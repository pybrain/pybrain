__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from scipy import ravel, zeros

from pybrain.rl.learners.learner import Learner
from pybrain.rl.explorers.continuous import NormalExplorer
from pybrain.utilities import abstractMethod
from pybrain.auxiliary import GradientDescent


class PolicyGradientLearner(Learner):
    """ The PolicyGradientLearner takes a ReinforcementDataSet and calculates the log likelihood
        for each parameter for each time step. The actual gradient estimation is done by
        subclasses.
    """
    
    defaultExploration = NormalExplorer
    
    def __init__(self):
        self.gd = GradientDescent()
                
    def setModule(self, module):
        Learner.setModule(self, module)
        # init the gradient descender with the module's parameters
        self.gd.init(module.params)    
    
    def learn(self):
        """ calls the gradient calculation function and executes a step in direction
            of the gradient, scaled with a small learning rate alpha. """
        assert self.dataset != None
        assert self.module != None
        
        # calculate the gradient with the specific function from subclass
        gradient = ravel(self._calculateGradient())

        # scale gradient if it has too large values
        if max(gradient) > 1000:
            gradient = gradient / max(gradient) * 1000
        
        # update the parameters in the module
        p = self.gd(gradient)
        self.module._setParameters(p)
        self.module.reset()
        
    
    def _calculateLogLH(self):
        loglh = zeros(len(self.dataset), len(self.module.params)) 


    def _calculateGradient(self):
        abstractMethod()