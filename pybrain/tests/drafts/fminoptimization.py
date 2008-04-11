from pybrain.rl.environments.functions import SphereFunction
from pybrain.rl.learners import NelderMead, CMAES


def testFMin():
    n = 2
    f = SphereFunction(n, xopt = [10]*n)
    E = NelderMead(f)
    print E.optimize()    
    
    
def testCMA():
    n = 2
    f = SphereFunction(n, xopt = [10]*n)
    E = CMAES(f)
    print E.optimize() 
    
    
if __name__ == '__main__':
    testFMin()
    testCMA()