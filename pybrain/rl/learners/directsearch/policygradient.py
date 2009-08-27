__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.rl.learners.learner import Learner
from pybrain.utilities import abstractMethod
from pybrain.auxiliary import GradientDescent
from pybrain.rl.explorers import NormalExplorer
from pybrain.datasets.dataset import DataSet

from scipy import ravel, array, zeros


class LoglhDataSet(DataSet):
    def __init__(self, dim):
        DataSet.__init__(self)
        self.addField('loglh', dim)
        self.linkFields(['loglh'])
        self.index = 0
        

class PolicyGradientLearner(Learner):
    """ PolicyGradientLearner is a super class for all continuous direct search
        algorithms that use the log likelihood of the executed action to update
        the weights. Subclasses are ENAC, GPOMDP, or REINFORCE.
    """
    
    _module = None
    
    def __init__(self):
        # create default explorer
        self.explorer = None
        
        # gradient descender
        self.gd_module = GradientDescent()
        self.gd_explorer = GradientDescent()
        
        # loglh datasets
        self.loglh_module = None
        self.loglh_explorer = None
    
    def _setLearningRate(self, alpha):
        """ pass the alpha value through to the gradient descent object """
        self.gd.alpha = alpha
    
    def _getLearningRate(self):
        return self.gd.alpha
    
    learningRate = property(_getLearningRate, _setLearningRate)
        
    def _setModule(self, module):
        """ initialize gradient descender with module parameters and
            the loglh dataset with the outdim of the module. """
        self._module = module
        
        # initialize explorer
        self.explorer = NormalExplorer(module.outdim)
        
        # initialize gradient descenders
        self.gd_module.init(module.params) 
        self.gd_explorer.init(self.explorer.params)
        
        # initialize loglh datasets
        self.loglh_module = LoglhDataSet(module.paramdim) 
        self.loglh_explorer = LoglhDataSet(self.explorer.paramdim)      
    
    def _getModule(self):
        return self._module
        
    module = property(_getModule, _setModule)   
        
    def learn(self):
        """ calls the gradient calculation function and executes a step in direction
            of the gradient, scaled with a small learning rate alpha. """
        assert self.dataset != None
        assert self.module != None
                
        # calculate the gradient with the specific function from subclass
        gradient_module, gradient_explorer = self.calculateGradients()

        # scale gradient if it has too large values
        if max(gradient_module) > 1000:
            gradient_module = gradient_module / max(gradient_module) * 1000
        
        # update the parameters of the module
        p = self.gd_module(gradient_module)
        self.module._setParameters(p)
        self.module.reset()
        
        # scale gradient if it has too large values
        if max(gradient_explorer) > 1000:
            gradient_explorer = gradient_explorer / max(gradient_explorer) * 1000
        
        # update the parameters of the explorer
        p = self.gd_explorer(gradient_explorer)
        self.explorer._setParameters(p)
        self.explorer.reset()
        
    
    def explore(self, state, action):
        # forward pass of exploration
        explorative = Learner.explore(self, state, action).copy()
        
        # backward pass and store derivs of explorer
        self.explorer.backward()
        self.loglh_explorer.appendLinked(self.explorer.derivs.copy())
        
        # propagate inerr of explorer through module and store derivs
        expl_inerr = self.explorer.inputerror[self.explorer.offset]
        self.module.backActivate(expl_inerr)
        self.loglh_module.appendLinked(self.module.derivs.copy())
        
        return explorative
    
    
    def reset(self):
        self.loglh_module.clear()
        self.loglh_explorer.clear()
    
    
    def calculateGradients(self):
        abstractMethod()