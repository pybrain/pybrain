""" Simple script to do individual tests on FEM """

__author__ = 'Daan Wierstra and Tom Schaul'

from scipy import randn, log10, array

from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import storeCallResults
from pybrain.rl.learners import CMAES, FEM
from pybrain.rl.tasks.polebalancing import CartPoleTask
from pybrain.rl.environments.functions import SphereFunction, RosenbrockFunction, OppositeFunction, CigarFunction, RotateFunction, TranslateFunction
from pybrain.tests.helpers import sortedProfiling
from pybrain.structure.connections.full import FullConnection
import pylab





if False:
    f = CartPoleTask(2, markov = False)
    x0 = buildNetwork(f.outdim, 3, f.indim, bias = False)
    x0.addRecurrentConnection(FullConnection(x0['hidden0'], x0['hidden0'], name = 'rec'))
    x0.sortModules()
    ff = FEM(f, x0,
            batchsize = 25, 
            onlineLearning = True,
            gini = 0.15,
            giniPlusX = 0.15,
            maxupdate = 0.02,
            elitist = False,
            superelitist = True,
            ranking = 'toplinear',
            topselection = 10,
            verbose = True,
            maxEvaluations = 10000,
            )
        

else:
    dim = 15
    basef = RosenbrockFunction(dim)
    f = TranslateFunction(RotateFunction(basef))
    x0 = randn(dim)
    ff = FEM(f, x0,
            batchsize = 100, 
            onlineLearning = True,
            gini = 0.02,
            giniPlusX = 0.15,
            maxupdate = 0.005,
            elitist = False,
            superelitist = True,
            ranking = 'toplinear',
            temperature = 10.0,
            topselection = 50,
            verbose = True,
            maxEvaluations = 100000,
            )
            
            
res = storeCallResults(f)




        

print ff.learn(), len(res)
if True:
    #pylab.plot(log10(-array(res)))
    pylab.plot(log10(-array(ff.muevals)))
    pylab.show()

#E = CMAES(OppositeFunction(f), silent=False)
#print E.optimize()
