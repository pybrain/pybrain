from __future__ import print_function

############################################################################
# PyBrain Tutorial "Black Box Optimization"
# 
# Author: Tom Schaul, tom@idsia.ch
############################################################################

__author__ = 'Tom Schaul, tom@idsia.ch'

""" A script that attempts to illustrate a large variety of 
use-cases for different kinds of black-box learning algorithms. 
"""

from pybrain.structure.networks.network import Network
from pybrain.optimization import * #@UnusedWildImport

""" The problem we would like to solve can be anything that 
has something like a fitness function. 
The following switches between two different examples. 

The variable 'theparams' contains the trainable 
parameters that affect the fitness. """

if False:
    """ Simple function optimization:
    here the parameters are learned directly. """    
    from scipy import randn
    from pybrain.rl.environments.functions import SphereFunction
    thetask = SphereFunction(3)
    theparams = randn(3)
    
else:
    """ Simple pole-balancing task:
    here we learn the weights of a neural network controller."""   
    from pybrain.tools.shortcuts import buildNetwork
    from pybrain.rl.environments.cartpole.balancetask import BalanceTask
    thetask = BalanceTask()
    theparams = buildNetwork(thetask.outdim, thetask.indim, bias=False)    
    
print('Subsequently, we attempt to solve the following task:')
print(thetask.__class__.__name__)


if isinstance(theparams, Network):
    print('\nby finding good weights for this (simple) network:')
    print(theparams)
    print('\nwhich has', theparams.paramdim, 'trainable parameters. (the dimensions of its layers are:', end=' ')
    for m in theparams.modules:
        print(m.indim, ',', end=' ')
    print(')\n')
        
""" We allow every algorithm a limited number of evaluations. """

maxEvals = 1000

""" Standard function minimization: """

print('fmin', NelderMead(thetask, theparams, maxEvaluations=maxEvals).learn())

""" The same, using some other algorithms 
(note that the syntax for invoking them is always the same) """

print('CMA', CMAES(thetask, theparams, maxEvaluations=maxEvals).learn())
print('NES', ExactNES(thetask, theparams, maxEvaluations=maxEvals).learn())
print('FEM', FEM(thetask, theparams, maxEvaluations=maxEvals).learn())
print('Finite Differences', FiniteDifferences(thetask, theparams, maxEvaluations=maxEvals).learn())
print('SPSA', SimpleSPSA(thetask, theparams, maxEvaluations=maxEvals).learn())
print('PGPE', PGPE(thetask, theparams, maxEvaluations=maxEvals).learn())

""" Evolutionary methods fall in the Learner framework as well. 
All the following are examples."""

print('HillClimber', HillClimber(thetask, theparams, maxEvaluations=maxEvals).learn())
print('WeightGuessing', WeightGuessing(thetask, theparams, maxEvaluations=maxEvals).learn())
print('ES 50+50', ES(thetask, theparams, maxEvaluations=maxEvals).learn())

""" We can change some default parameters, e.g."""
print('ES 5+5', ES(thetask, theparams, mu=5, lambada=5, maxEvaluations=maxEvals).learn())

""" Memetic algorithms are a kind of meta-algorithm, doing topology mutations 
on the top-level, and using other algorithms internally 
as a kind of local search (default there: hill-climbing)."""

print('Memetic Climber', MemeticSearch(thetask, theparams, maxEvaluations=maxEvals).learn())
print('Memetic ES 50+50', MemeticSearch(thetask, theparams, maxEvaluations=maxEvals,
                                        localSearch=ES, localSteps=200).learn())
print('Memetic ES 5+5', MemeticSearch(thetask, theparams, maxEvaluations=maxEvals,
                                      localSearch=ES,
                                      localSearchArgs={'mu': 5, 'lambada': 5}).learn())
print('Memetic NES', MemeticSearch(thetask, theparams, maxEvaluations=maxEvals,
                                   localSearch=ExactNES,
                                   localSearchArgs={'batchSize': 20}).learn())

""" Inner memetic is the population based variant (on the topology level). """

print('Inner Memetic Climber', InnerMemeticSearch(thetask, theparams, maxEvaluations=maxEvals).learn())
print('Inner Memetic CMA', InnerMemeticSearch(thetask, theparams, maxEvaluations=maxEvals,
                                              localSearch=CMAES).learn())

""" Inverse memetic algorithms do local search on topology mutations, 
and weight changes in the top-level search. """

print('Inverse Memetic Climber', InverseMemeticSearch(thetask, theparams, maxEvaluations=maxEvals).learn())

