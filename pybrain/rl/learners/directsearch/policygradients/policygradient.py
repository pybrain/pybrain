__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.rl.learners.directsearch.directsearch import DirectSearch
from pybrain.rl.learners.datasetlearner import DataSetLearner
from pybrain.utilities import abstractMethod
from pybrain.auxiliary import GradientDescent
from scipy import ravel

class PolicyGradientLearner(DirectSearch, DataSetLearner):
    """ The PolicyGradientLearner takes a ReinforcementDataSet which has been extended with the log likelihood
        of each parameter for each time step. It additionally takes a Module which has been extended with
        a gaussian layer on top. It then changes the weights of both the gaussian layer and the rest of the
        module according to a specific gradient descent, which is a derivation of this base class.
    """
    def __init__(self):
        self.gd = GradientDescent()
        
    def setAlpha(self, alpha):
        """ pass the alpha value through to the gradient descent object """
        self.gd.alpha = alpha
        print "patching through to gd", alpha
    
    def getAlpha(self):
        return self.gd.alpha
    
    alpha = property(getAlpha, setAlpha)
        
    def setModule(self, module):
        DataSetLearner.setModule(self, module)
        self.gd.init(module.params)    
        
    def learn(self):
        """ calls the gradient calculation function and executes a step in direction
            of the gradient, scaled with a small learning rate alpha. """
        assert self.ds != None
        assert self.module != None
        
        # calculate the gradient with the specific function from subclass
        gradient = ravel(self.calculateGradient())

        # scale gradient if it has too large values
        if max(gradient) > 1000:
            gradient = gradient / max(gradient) * 1000
        
        # update the parameters
        p = self.gd(gradient)
        self.module._setParameters(p)
        self.module.reset()
        
    def calculateGradient(self):
        abstractMethod()