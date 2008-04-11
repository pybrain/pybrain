__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.environments.functions import SphereFunction, OppositeFunction
from pybrain.rl.learners import LiMaG, CMAES


def testLimag():
    n = 10
    f = OppositeFunction(SphereFunction(n, xopt = [1]*n))
    f.desiredValue = -0.1
    E = LiMaG(f)
    print E.optimize()    
    print len(f.xlist)
    
    
def testCMA():
    n = 10
    f = SphereFunction(n, xopt = [1]*n)
    f.desiredValue = 0.1
    E = CMAES(f)
    print E.optimize() 
    print len(f.xlist)
    
    
if __name__ == '__main__':
    testCMA()
    testLimag()
    