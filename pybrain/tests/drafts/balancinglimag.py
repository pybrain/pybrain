""" Try LiMaG and CMA on the cart-pole task """

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.environments.functions.episodicevaluators import CartPoleEvaluator
from pybrain.rl.environments.functions import OppositeFunction
from pybrain.rl.learners import CMAES
from pybrain.rl.learners.blackboxoptimizers.evolution.limag import LiMaG
from pybrain.tools.shortcuts import buildSimpleNetwork


def testLimag():
    m = buildSimpleNetwork(4, 1, 1, True)
    f = CartPoleEvaluator(m)
    f.desiredValue = 500
    E = LiMaG(f)
    E.minimize = False
    print E.optimize()    
    print len(f.xlist)
    
    
def testCMA():
    m = buildSimpleNetwork(4, 1, 1, True)
    f = OppositeFunction(CartPoleEvaluator(m))
    f.desiredValue = -500
    E = CMAES(f, silent = False)
    print E.optimize() 
    print len(f.xlist)
    
    
if __name__ == '__main__':
    testCMA()
    testLimag()
    