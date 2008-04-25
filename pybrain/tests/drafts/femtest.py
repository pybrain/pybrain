""" Simple script to do individual tests on FEM """

__author__ = 'Daan Wierstra and Tom Schaul'

from scipy import randn, log10, array

from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import storeCallResults
from pybrain.rl.learners import CMAES, FEM
from pybrain.rl.tasks.polebalancing import CartPoleTask
from pybrain.rl.environments.functions import SphereFunction, RosenbrockFunction, SchwefelFunction, OppositeFunction, CigarFunction, RotateFunction, TranslateFunction
from pybrain.tests.helpers import sortedProfiling
from pybrain.structure.connections.full import FullConnection
import pylab





if True:
    f = CartPoleTask(2, markov = False)
    x0 = buildNetwork(f.outdim, 3, f.indim, bias = False)
    x0.addRecurrentConnection(FullConnection(x0['hidden0'], x0['hidden0'], name = 'rec'))
    x0.sortModules()
    ff = FEM(f, x0,
            batchsize = 50, 
            onlineLearning = True,
            gini = 0.15,
            giniPlusX = 0.15,
            maxupdate = 0.05,
            elitist = False,
            superelitist = True,
            ranking = 'toplinear',
            topselection = 5,
            verbose = True,
            maxEvaluations = 10000,
            )
        

else:
    dim = 15
    basef = SphereFunction(dim)
    f = TranslateFunction(RotateFunction(basef))
    x0 = randn(dim)
    ff = FEM(f, x0,
            batchsize = 50, 
            onlineLearning = True,
            gini = 0.15,
            giniPlusX = 0.15,
            maxupdate = 0.03,
            elitist = False,
            superelitist = False,
            ranking = 'toplinear',
            temperature = 10.0,
            topselection = 7,
            unlawfulExploration = 1.0,
            verbose = True,
            maxEvaluations = 200000,
            )
            
            
res = storeCallResults(f)




        

print ff.learn(), len(res)
if True:
    #pylab.plot(log10(-array(res)))
    pylab.plot(log10(-array(ff.muevals)))
    pylab.show()

#E = CMAES(OppositeFunction(f), silent=False)
#print E.optimize()
