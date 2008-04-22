""" A script that attempts to *illustrate* a large variety of use-cases for Learners """

__author__ = 'Tom Schaul, tom@idsia.ch'


from pybrain import buildNetwork, Network
from pybrain.utilities import storeCallResults
from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.structure.connections.full import FullConnection
from pybrain.structure.evolvables.cheaplycopiable import CheaplyCopiable

from pybrain.rl.learners import NelderMead, CMAES, NaturalEvolutionStrategies, HillClimber 
from pybrain.rl.learners import ES, FEM, EpisodicRL, Reinforce, WeightGuessing
from pybrain.rl.learners.finitedifference import FDBasic, SPLA
from pybrain.rl.learners.meta import MemeticSearch, InnerMemeticSearch
from pybrain.rl.agents.finitedifference import FiniteDifferenceAgent
from pybrain.rl.tasks.episodic import EpisodicTask

from pybrain.rl.environments.functions import SphereFunction
from pybrain.rl.tasks.polebalancing import CartPoleTask
from pybrain.rl.tasks.pomdp import CheeseMaze


# TODO: a simple clean interface for making noisy evaluators average their results


def main(maxEvals = 500):
    if False:
        # simple function optimization
        thetask = SphereFunction(3)
        thenet = ParameterContainer(3)
        
    elif True:
        # simple pole-balancing task
        thetask = CartPoleTask(1, markov = True)
        thenet = buildNetwork(thetask.outdim, thetask.indim, bias = False)
        
    elif True:
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
    res = storeCallResults(thetask)
    
    
    
    print 'Subsequently, we attempt to solve the task:'
    print thetask
    
    if isinstance(thenet, Network):
        print '\nby finding good weights for this (simple) network:'
        print thenet
        print '\nwhich has', thenet.paramdim, 'trainable parameters. (the dimensions of its layers are:',
        for m in thenet.modules:
            print m.indim, ',',
        print ')\n'
            
    
    # ---- standard FMIN ---
    print 'fmin', NelderMead(thetask, thenet).learn(maxEvals), len(res)
    
    # the same, using other algorithms
    print 'CMA', CMAES(thetask, thenet).learn(maxEvals), len(res)
    print 'NES', NaturalEvolutionStrategies(thetask, thenet).learn(maxEvals), len(res)
    print 'FEM', FEM(thetask, thenet).learn(maxEvals), len(res)
    
    
    # ----  RL ----
    if isinstance(thetask, EpisodicTask):
        print 'Episodic NAC', EpisodicRL(thetask, thenet).learn(maxEvals), len(res)
        print 'REINFORCE', EpisodicRL(thetask, thenet,
                                      sublearner = Reinforce).learn(maxEvals), len(res)
        print 'Finite Differences', EpisodicRL(thetask, thenet, subagent = FiniteDifferenceAgent, 
                                               sublearner = FDBasic).learn(maxEvals), len(res)    
        print 'SPLA', EpisodicRL(thetask, thenet, subagent = FiniteDifferenceAgent, 
                                 sublearner = SPLA).learn(maxEvals), len(res)    
    
    
    # ---- evolutionary methods ----
    # for all these we want network copies to be cheap, thus:
    thenet = CheaplyCopiable(thenet)
    
    print 'HillClimber', HillClimber(thetask, thenet).learn(maxEvals), len(res)
    print 'WeightGuessing', WeightGuessing(thetask, thenet).learn(maxEvals), len(res)
    print 'ES 50+50', ES(thetask, thenet).learn(maxEvals), len(res)
    print 'ES 5+5', ES(thetask, thenet, mu = 5, lambada = 5).learn(maxEvals), len(res)
    
    # memetic with different types of local search
    print 'Memetic Climber', MemeticSearch(thetask, thenet).learn(maxEvals), len(res)
    print 'Memetic CMA', MemeticSearch(thetask, thenet, CMAES).learn(maxEvals), len(res)
    print 'Memetic ES 50+50', MemeticSearch(thetask, thenet, ES).learn(maxEvals), len(res)
    print 'Memetic ES 5+5', MemeticSearch(thetask, thenet, ES, 
                                          localSearchArgs = {'mu': 5, 'lambada': 5}).learn(maxEvals), len(res)
    
    # inner memetic
    print 'Inner Memetic Climber', InnerMemeticSearch(thetask, thenet).learn(maxEvals), len(res)
    print 'Inner Memetic CMA', InnerMemeticSearch(thetask, thenet, CMAES).learn(maxEvals), len(res)
    
    # inverse memetic
    
    
    return res
    
if __name__ == '__main__':
    import pylab
    pylab.plot(main())
    #pylab.savefig('tmp.eps')
    pylab.show()
