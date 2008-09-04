############################################################################
# PyBrain Tutorial "Learners - Black Box Optimization"
# 
# Author: Tom Schaul, tom@idsia.ch
############################################################################

__author__ = 'Tom Schaul, tom@idsia.ch'

""" A script that attempts to illustrate a large variety of use-cases for Learners """

from pybrain import buildNetwork
from pybrain.utilities import storeCallResults
from pybrain.structure import FullConnection
from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.structure.evolvables.cheaplycopiable import CheaplyCopiable
from pybrain.structure.networks.network import Network
from pybrain.rl.learners import * #@UnusedWildImport
from pybrain.rl.learners.meta import * #@UnusedWildImport
from pybrain.rl.agents.finitedifference import FiniteDifferenceAgent
from pybrain.rl.tasks.episodic import EpisodicTask
from pybrain.rl.environments.functions import SphereFunction
from pybrain.rl.tasks.polebalancing import CartPoleTask
from pybrain.rl.tasks.pomdp import CheeseMaze

""" The problem we we would like to solve can be anything that has something like a fitness
function. The following switches between 4 different examples. 

The variable 'thenet' contains the trainable parameters that affect the fitness. """

if False:
    # simple function optimization
    thetask = SphereFunction(3)
    thenet = ParameterContainer(3)
    
elif True:
    # simple pole-balancing task
    thetask = CartPoleTask(1, markov = True)
    thenet = buildNetwork(thetask.outdim, thetask.indim, bias = False)
    
elif False:
    # hard pole-balancing task
    thetask = CartPoleTask(2, markov = False)
    thenet = buildNetwork(thetask.outdim, 3, thetask.indim)
    thenet.addRecurrentConnection(FullConnection(thenet['hidden0'], thenet['hidden0']))
    thenet.sortModules()
    
elif False:
    # maze-navigation 
    # TODO: noisy task, calls should be averaged!
    thetask = CheeseMaze(maxSteps = 50)
    thenet = buildNetwork(thetask.outdim, 1, thetask.indim)
    
print 'Subsequently, we attempt to solve the following task:'
print thetask

if isinstance(thenet, Network):
    print '\nby finding good weights for this (simple) network:'
    print thenet
    print '\nwhich has', thenet.paramdim, 'trainable parameters. (the dimensions of its layers are:',
    for m in thenet.modules:
        print m.indim, ',',
    print ')\n'
        
""" We store all the evaluations of the task in a list: """
res = storeCallResults(thetask)

""" We allow every algorithm a limited number of evaluations. """

maxEvals = 1000

""" Standard function minimization: """

print 'fmin', NelderMead(thetask, thenet).learn(maxEvals)


""" the same, using other algorithms """

print 'CMA', CMAES(thetask, thenet).learn(maxEvals)
print 'NES', NaturalEvolutionStrategies(thetask, thenet).learn(maxEvals)
print 'FEM', FEM(thetask, thenet).learn(maxEvals)


""" if the task can be framed as a RL problem, we can use those algorithms: """

if isinstance(thetask, EpisodicTask):
    print 'Episodic NAC', EpisodicRL(thetask, thenet).learn(maxEvals)
    print 'REINFORCE', EpisodicRL(thetask, thenet,
                                  sublearner = Reinforce).learn(maxEvals)
    print 'Finite Differences', EpisodicRL(thetask, thenet, subagent = FiniteDifferenceAgent, 
                                           sublearner = FDBasic).learn(maxEvals)   
    print 'SPLA', EpisodicRL(thetask, thenet, subagent = FiniteDifferenceAgent, 
                             sublearner = SPLA).learn(maxEvals)   


""" Evolutionary methods fall in the Learner framework as well. 
All the following are examples.

Note that for all these we want network copies to be cheap, thus: """
thenet = CheaplyCopiable(thenet)

print 'HillClimber', HillClimber(thetask, thenet).learn(maxEvals)
print 'WeightGuessing', WeightGuessing(thetask, thenet).learn(maxEvals)
print 'ES 50+50', ES(thetask, thenet).learn(maxEvals)
print 'ES 5+5', ES(thetask, thenet, mu = 5, lambada = 5).learn(maxEvals)

""" memetic algorithms are a kind of meta-algorithm that can use other Learners 
as a kind of local search """

print 'Memetic Climber', MemeticSearch(thetask, thenet).learn(maxEvals)
print 'Memetic CMA', MemeticSearch(thetask, thenet, CMAES).learn(maxEvals)
print 'Memetic ES 50+50', MemeticSearch(thetask, thenet, ES).learn(maxEvals)
print 'Memetic ES 5+5', MemeticSearch(thetask, thenet, ES, 
                                      localSearchArgs = {'mu': 5, 'lambada': 5}).learn(maxEvals)

""" Inner memetic is the population based variant """

print 'Inner Memetic Climber', InnerMemeticSearch(thetask, thenet).learn(maxEvals)
print 'Inner Memetic CMA', InnerMemeticSearch(thetask, thenet, CMAES).learn(maxEvals)

""" Inverse memetic algorithms do local search on topology mutations, and have weight changes in the outer search. """

print 'Inverse Memetic Climber', InverseMemeticSearch(thetask, thenet).learn(maxEvals)
