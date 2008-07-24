__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.environments.functions import SphereFunction, RosenbrockFunction, OppositeFunction
from pybrain.rl.learners.blackboxoptimizers.nes2 import NaturalEvolutionStrategies2

from scipy import ones

def testFunc():
    n = 2
    return SphereFunction(n, xopt = [-1]*n)
    #return RosenbrockFunction(n, xopt = [-1]*n)
    #return OppositeFunction(SphereFunction(n, xopt = [-1]*n))
    #return OppositeFunction(RosenbrockFunction(n, xopt = [0]*n))

def testNES():
    f = testFunc()
    E = NaturalEvolutionStrategies2(f, ones(2),
                                    learningRate = 0.001,
                                    batchSize = 10)                                   
    print E.learn(30000)
    
if __name__ == '__main__':
    testNES()
    #testCMA()