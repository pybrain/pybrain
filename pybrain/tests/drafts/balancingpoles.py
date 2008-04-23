""" Try CMA on all the cart-pole tasks """

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.environments.cartpole import CartPoleEnvironment, DoublePoleEnvironment, NonMarkovPoleEnvironment, NonMarkovDoublePoleEnvironment
from pybrain.rl.environments.functions.episodicevaluators import CartPoleEvaluator, EpisodicEvaluator
from pybrain.rl.environments.functions import OppositeFunction
from pybrain import FullConnection
from pybrain.rl.learners.blackboxoptimizers.fem import FEM
from pybrain.rl.learners import CMAES
from pybrain import buildNetwork
from pybrain.rl.tasks.polebalancing import CartPoleTask
from pybrain.structure.modules.lstm import LSTMLayer
from scipy import ravel
    
def testBalancing(env):
    # a network of the correct size, with a bias, some hidden neurons 
    if isinstance(env, DoublePoleEnvironment):
        hidden = 4
    else:
        hidden = 1            
    m = buildNetwork(env.outdim, hidden, env.indim)
    if isinstance(env, NonMarkovPoleEnvironment):
        # add recurrent connections in the pomdp case
        m.addRecurrentConnection(FullConnection(m['h'], m['h'], name = 'rec'))
        m.sortModules()
        
    print m
    print 'nb of parameters:', m.paramdim
    
    # invert the fitness function, so that CMA can minimize it
    f = OppositeFunction(CartPoleEvaluator(m, env))
    f.desiredValue = -500
        
    E = CMAES(f, silent = False, maxEvals = 5000)
    print E.optimize() 
    print len(f.xlist)


def testOtherBalancing():
    markov = False
    t = CartPoleTask(numPoles = 2, markov = markov, extraObservations = True)
    net = buildNetwork(t.getOutDim(), 3, t.getInDim(), bias = False)#, hiddenclass = LSTMLayer)
    if not markov:
        # add recurrent connections in the pomdp case
        net.addRecurrentConnection(FullConnection(net['hidden0'], net['hidden0'], name = 'rec'))
        net.sortModules()
    net.params *= 10
    
    #f = OppositeFunction(EpisodicEvaluator(net, t))
    #f.desiredValue = -50000
    profiling = False
    f = EpisodicEvaluator(net, t)
    f.desiredValue = 5000
        
    #E = CMAES(f, silent = False, maxEvals = 50000)
    global E
    E = FEM(f, maxEvals = 2000)
    if profiling:
        from pybrain.tests.helpers import sortedProfiling
        sortedProfiling('E.optimize()')
    else:
        print 'final weights', ravel(E.optimize())
        print 'number of evaluations', len(f.xlist)
    
if __name__ == '__main__':
    testOtherBalancing()
    #testBalancing(CartPoleEnvironment())
    #testBalancing(DoublePoleEnvironment())
    #testBalancing(NonMarkovPoleEnvironment())
    #testBalancing(NonMarkovDoublePoleEnvironment())
    