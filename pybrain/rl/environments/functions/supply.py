from pybrain.utilities import setAllArgs
__author__ = 'Hobson Lane <hobson@totalgood.com> and Tom Schaul, tom@idsia.ch'

from scipy import zeros, array, ndarray

# from pybrain.rl.environments import Environment
from pybrain.rl.environments.functions.function import FunctionEnvironment 
from pybrain.structure.parametercontainer import ParameterContainer
# from pybrain.rl.environments.fitnessevaluator import FitnessEvaluator


class SupplyEnvironment(FunctionEnvironment):
    """ N-to-1 mapping with a single goal (default=zero) when the model parameters (x) are optimal (xopt) . 

    Useful for optimally supplying a valuable, but regularly replenished resource to a demand:
        - water to a farm
        - power to a building
        - funds to an equity trader

    The supply and demand time-series is provided as the first 2 columns in a dataframe.
    The supply value is a incremental change or influx of the resource (e.g. Joules of energy or gallons of water).
    The demand value is an incremental outflow of the resource (Joules of energy or gallons of water). 
    Divided by the sample period, the demand represents a flow rate (Watts of power or gallons/second).
    The performance period is a number of seconds after which the performance metric is reset.
    The performance metric for this FunctionEnvironment is the residual unsupplied resource above peak demand.

    So if your resource is water from an on-site water tank and you want to minimize the maximum flow rate from the 
    utility company (to refill your tank and perform your business), this algorithm will turn on the tank spigot
    optimally (once it has been trained, and presuming your business needs are predictable).

    """

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

    def __init__(self, supply_demand_dataframe, performance_reset_period='month', xdim=None, xopt=None, xbound=1, feasible=True, constrained=False, violation=False, **args):

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
        """ The one sensor is the function result. """
        tmp = self.result
        assert tmp is not None
        self.result = None
        return array([tmp])

    def performAction(self, action):
        """ The action argument is a numpy array of values for the function inputs. 

        Activate
        """
        self.result = self(action)

    @property
    def indim(self):
        return self.xdim

    # does not provide any observations
    outdim = 0

