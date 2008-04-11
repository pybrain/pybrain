__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.environments.functions import SphereFunction, RosenbrockFunction, OppositeFunction
from pybrain.rl.learners import NaturalEvolutionStrategies, CMAES


def testFunc():
    n = 15
    #return SphereFunction(n, xopt = [0]*n)
    return OppositeFunction(RosenbrockFunction(n, xopt = [0]*n))

def testNES():
    f = testFunc()
    E = NaturalEvolutionStrategies(f, 
                                   #lr = 0.0002, lrSigma=0.00005,
                                   lr = 0.002, #lrSigma = 0.0005,#slidingbatch
                                   #momentum = 0.0,
                                   #lr = 0.02,
                                   #lr = 0.00002,
                                   #lr = 1.1, lrPeters = True,#epsilon=0.00001,
                                   #lr = 1.5, lrPeters=True, 
                                   #onefifth = True, onefifthAdaptation = 1.5,
                                   mu = 1, #ridge = False, ridgeconstant = 1e-3,
                                   lambd = int((15*15+15)*2.5), stopPrecision = 1e-10, gini=0.05,
                                   ranking = 'smooth', slidingbatch = False, importanceSampling = False)
    print E.optimize()    
    
    
def testCMA():
    f = OppositeFunction(testFunc())
    E = CMAES(f, silent=False)
    print E.optimize() 
    print len(f.xlist)
    
    
if __name__ == '__main__':
    testNES()
    #testCMA()