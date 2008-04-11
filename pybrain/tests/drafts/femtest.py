__author__ = 'Daan Wierstra and Tom Schaul'

from pybrain.rl.learners.blackboxoptimizers.fem import FEM
from pybrain.rl.learners import CMAES

from pybrain.rl.environments.functions import SphereFunction, RosenbrockFunction, OppositeFunction, CigarFunction

f= OppositeFunction(CigarFunction(50))
ff = FEM(f)
ff.optimize()
#E = CMAES(OppositeFunction(f), silent=False)
#print E.optimize()

