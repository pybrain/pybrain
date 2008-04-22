__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.rl.learners.rllearner import RLLearner
from pybrain.utilities import abstractMethod
from pybrain.auxiliary import GradientDescent
from scipy import ravel, clip

class PolicyGradientLearner(RLLearner):
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
        #print "patching through to gd"
    
    def getAlpha(self, alpha):
        return self.gd.alpha
    
    alpha = property(getAlpha, setAlpha)
        
    def setModule(self, module):
        RLLearner.setModule(self, module)
        self.gd.init(module.params)      
        
    def learn(self):
        """ calls the gradient calculation function and executes a step in direction
            of the gradient, scaled with a small learning rate alpha. """
        assert self.ds != None
        assert self.module != None
        
        # calculate the gradient with the specific function from subclass
        gradient = ravel(self.calculateGradient())
        # prevent gradient having too large values
        gradient = clip(gradient, -50, +50)
        # update the parameters
        p = self.gd(gradient)
        self.module._setParameters(p)
        self.module.reset()
        
    def calculateGradient(self):
        abstractMethod()