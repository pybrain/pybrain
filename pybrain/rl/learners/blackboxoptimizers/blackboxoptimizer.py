__author__ = 'Tom Schaul, tom@idsia.ch'

from numpy.random import randn

from pybrain.utilities import abstractMethod
from pybrain.rl.environments.functions import FunctionEnvironment
from pybrain.rl.learners import Learner


class BlackBoxOptimizer(Learner):
    """ a class capable of finding a minimal/maximal value of an unknown (single-output) function. """

    # the function must take an array as input and produce a single value
    targetfun = None
    tfun = None # this is keeping the TestFunction object, if applicable
    
    # dimensionality of the input
    xdim = 1

    # minimize or maximize?
    minimize = True
    
    # the best first guess (should be an array)
    x0 = None
    
    # maximal number of function evaluations
    maxEvals = 1e6
    
    # stopping criterion
    stopPrecision = 1e-10   
    
    def __init__(self, f, **parameters):
        """ make sure everything is initialized correctly, and that the parameters are consistent. """
        self.setArgs(**parameters)
        
        assert isinstance(f, FunctionEnvironment)
        self.tfun = f
        if 'stopPrecision' not in parameters:
            self.stopPrecision = f.desiredValue                        
        f.reset()
        if self.xdim == 1:
            # take over the given dimension
            self.xdim = f.xdim
        assert self.xdim == f.xdim
        self.targetfun = lambda x: f.controlledExecute(x)
            
        if self.x0 == None:
            self.x0 = randn(self.xdim)
    
    def optimize(self):
        """ core method, to be implemented by subclasses """    
        abstractMethod()
        
        