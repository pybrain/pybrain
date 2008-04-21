__author__ = 'Daan Wierstra and Tom Schaul'

from pybrain.rl.learners.blackboxoptimizers.fem import FEM
from pybrain.rl.learners import CMAES

from pybrain.rl.environments.functions import SphereFunction, RosenbrockFunction, OppositeFunction, CigarFunction

f= OppositeFunction(RosenbrockFunction(15))
ff = FEM(f, batchsize = 100, onlineLearning = True,
        gini = 0.02,
        giniPlusX = 0.15,
        unlawfulExploration = 1.0,
        maxupdate = 0.1,
        elitist = False,
        superelitist = False)
ff.optimize()
E = CMAES(OppositeFunction(f), silent=False)
#print E.optimize()

