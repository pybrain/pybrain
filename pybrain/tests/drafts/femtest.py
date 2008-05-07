""" Simple script to do individual tests on FEM """
from pybrain.tools.rankingfunctions import TopLinearRanking
from pybrain.tools.rankingfunctions import ExponentialRanking

__author__ = 'Daan Wierstra and Tom Schaul'

from scipy import randn, log10, array, eye

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
    net = buildNetwork(f.outdim, 3, f.indim, bias = False)
    net.addRecurrentConnection(FullConnection(net['hidden0'], net['hidden0'], name = 'rec'))
    net.sortModules()
    ff = FEM(f, net,
             batchsize = 50, 
             onlineLearning = True,
             forgetFactor = 0.05,
             useCauchy = True,
             elitist = True,
             rankingFunction = TopLinearRanking(topFraction = 0.3),
             #rankingFunction = ExponentialRanking(temperature = 10),
             verbose = True,
             maxEvaluations = 10000,
             #initCovariances = eye(net.paramdim)*0.01
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
