__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import zeros, array, ndarray

from pybrain.rl.environments import Environment
from pybrain.utilities import abstractMethod
from pybrain.rl.evaluator import Evaluator
from pybrain.structure.parametercontainer import ParameterContainer


class FunctionEnvironment(Environment, Evaluator):
    """ A n-to-1 mapping function to be with a single minimum of value zero, at xopt. """
    
    # what input dimensions can the function have?
    xdimMin = 1
    xdimMax = None
    xdim = None
    
    # the (single) point where f = 0
    xopt = None
    
    # what would be the desired performance?
    desiredValue = -1e-10
    
    def __init__(self, xdim = 1, xopt = None):
        assert xdim >= self.xdimMin and not (self.xdimMax != None and xdim > self.xdimMax)
        self.xdim = xdim
        if xopt == None:
            self.xopt = zeros(self.xdim)
        else:
            self.xopt = xopt
        self.reset()
        
    def f(self, x):
        """ the function itself, to be defined by subclasses """
        abstractMethod()

    def __call__(self, x):
        """ the f(x) is to be minimized, but evaluators assume that 
        goal is maximization, so we negate the result here"""
        if isinstance(x, ParameterContainer):
            x = x.params
        assert type(x) == ndarray
        return -self.f(x-self.xopt)
    
    # methods for conforming to the Environment interface:
    def reset(self):
        self.result = None
        
    def getSensors(self):
        """ the one sensor is the function result. """
        tmp = self.result
        assert tmp != None
        self.result = None
        return array([tmp])
                    
    def performAction(self, action):
        """ the action is an array of values for the function """
        self.result = self(action)
    
    @property
    def indim(self):
        return self.xdim
    
    outdim = 1
    
    
class OppositeFunction(FunctionEnvironment):
    """ the opposite of a function """
    def __init__(self, basef):
        FunctionEnvironment.__init__(self, basef.xdim, basef.xopt)
        self.f = lambda x: -basef.f(x)
        