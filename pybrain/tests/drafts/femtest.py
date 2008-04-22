""" Simple script to do individual tests on FEM """

__author__ = 'Daan Wierstra and Tom Schaul'

from scipy import randn, log10, array
from pybrain.utilities import storeCallResults
from pybrain.rl.learners import CMAES, FEM
from pybrain.rl.tasks.polebalancing import CartPoleTask
from pybrain.rl.environments.functions import SphereFunction, RosenbrockFunction, OppositeFunction, CigarFunction, RotateFunction, TranslateFunction
from pybrain.tests.helpers import sortedProfiling
import pylab

dim = 15
basef = RosenbrockFunction(dim)

f = TranslateFunction(RotateFunction(basef))
res = storeCallResults(f)

ff = FEM(f, randn(dim), 
         batchsize = 100, 
         onlineLearning = True,
         gini = 0.02,
         giniPlusX = 0.15,
         unlawfulExploration = 1.0,
         maxupdate = 0.1,
         elitist = False,
         superelitist = False,
         verbose = True,
         maxEvaluations = 2000,
         )


print ff.learn(), len(res)
if True:
    #pylab.plot(log10(-array(res)))
    pylab.plot(log10(-array(ff.muevals)))
    pylab.show()
