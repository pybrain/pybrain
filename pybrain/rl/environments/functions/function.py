from pybrain.utilities import setAllArgs
__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import zeros, array, ndarray

from pybrain.rl.environments import Environment
from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.rl.environments.fitnessevaluator import FitnessEvaluator


class FunctionEnvironment(Environment, FitnessEvaluator):
    """ A n-to-1 mapping function to be with a single minimum of value zero, at xopt. """

    # what input dimensions can the function have?
    xdimMin = 1
    xdimMax = None
    xdim = None

    # the (single) point where f = 0
    xopt = None

    # what would be the desired performance? by default: something close to zero
    desiredValue = 1e-10
    toBeMinimized = True
    
    # does the function already include a penalization term, to keep search near the origin?
    penalized = False

    def __init__(self, xdim = None, xopt = None, xbound=5, feasible=True, constrained=False, violation=False, **args):
        self.feasible=feasible
        self.constrained=constrained
        self.violation=violation
        self.xbound=xbound
        if xdim is None:
            xdim = self.xdim
        if xdim is None:
            xdim = self.xdimMin
        assert xdim >= self.xdimMin and not (self.xdimMax is not None and xdim > self.xdimMax)
        self.xdim = xdim
        if xopt is None:
            self.xopt = zeros(self.xdim)
        else:
            self.xopt = xopt
        setAllArgs(self, args)
        self.reset()

    def __call__(self, x):
        if isinstance(x, ParameterContainer):
            x = x.params
        assert type(x) == ndarray, 'FunctionEnvironment: Input not understood: '+str(type(x))
        return self.f(x)

    # methods for conforming to the Environment interface:
    def reset(self):
        self.result = None

    def getSensors(self):
        """ the one sensor is the function result. """
        tmp = self.result
        assert tmp is not None
        self.result = None
        return array([tmp])

    def performAction(self, action):
        """ the action is an array of values for the function """
        self.result = self(action)

    @property
    def indim(self):
        return self.xdim

    # does not provide any observations
    outdim = 0

