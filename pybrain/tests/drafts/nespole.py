__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.environments.functions import SphereFunction, RosenbrockFunction, OppositeFunction
from pybrain.rl.learners import NaturalEvolutionStrategies, CMAES
from pybrain.rl.environments.cartpole import DoublePoleEnvironment
from pybrain.rl.environments.functions.episodicevaluators import CartPoleEvaluator
from pybrain import buildNetwork

def testFunc():
    n =3
    #return SphereFunction(n, xopt = [0]*n)
    #return RosenbrockFunction(n, xopt = [0]*n)
    env = DoublePoleEnvironment()
    n = buildNetwork(env.outdim, 2, env.indim)
    #print n, n.outdim, n.indim
    f=  CartPoleEvaluator(n, env)
    f.desiredValue = 100000
    return f
    
def testNES():
    f = testFunc()
    
    E = NaturalEvolutionStrategies(f, 
                                   lr = 0.01,
                                   #lr = 0.00005, #slidingbatch dim 10
                                   #momentum = 0.0,
                                   #lr = 0.02,
                                   #lr = 0.00002,
                                   #lr = 1.1, lrPeters = True,#epsilon=0.00001,
                                   #lr = 1.5, lrPeters=True, 
                                   #onefifth = True, onefifthAdaptation = 1.5,
                                   mu = 3,
                                   #maxEvals = 10,
                                   lambd = 200*2, gini=0.1,
                                   ranking = 'smooth', slidingbatch = False, importanceSampling = False)
    print E.optimize()    
    
    
def testCMA():
    f = testFunc()
    E = CMAES(f, silent=False)
    print E.optimize() 
    print len(f.xlist)
    
    
if __name__ == '__main__':
    testNES()
    #testCMA()