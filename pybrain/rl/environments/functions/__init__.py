from pybrain.rl.environments.functions.function import FunctionEnvironment
from pybrain.rl.environments.functions.unimodal import SchwefelFunction, SphereFunction, TabletFunction, DiffPowFunction, \
    CigarFunction, ElliFunction, RosenbrockFunction
from pybrain.rl.environments.functions.multimodal import RastriginFunction, AckleyFunction, GriewankFunction, Schwefel_2_13Function, \
    WeierstrassFunction, FunnelFunction
from pybrain.rl.environments.functions.unbounded import ParabRFunction, SharpRFunction, LinearFunction
from pybrain.rl.environments.functions.transformations import oppositeFunction, RotateFunction, TranslateFunction