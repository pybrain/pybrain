from scipy import ravel

from pybrain.rl import CMAES
from pybrain.tools.shortcuts import buildNetwork
from pybrain.rl.environments.simple import SimpleEnvironment, MinimizeTask
from pybrain.rl.environments.cartpole import BalanceTask, CartPoleEnvironment, DoublePoleEnvironment, NonMarkovPoleEnvironment, NonMarkovDoublePoleEnvironment
from pybrain.rl.environments.functions import RosenbrockFunction, OppositeFunction
from pybrain.rl.environments.functions.episodicevaluators import EpisodicEvaluator, CartPoleEvaluator
from pybrain.rl.learners.searchprocesses.memeticcmaclimber import MemeticCMAClimber
from pybrain.rl.evolvables import MaskedParameters


#t = MinimizeTask(SimpleEnvironment())
t = BalanceTask(CartPoleEnvironment())
#t = TrivialMaze()


hidden = 3
bias = True
n = buildNetwork(t.getOutDim(), hidden, t.getInDim(), bias = bias)
#mn = MaskedParameters(n, maskFlipProbability = 0.2)


#cl = MemeticCMAClimber(mn, t, localSteps = 20)

ee = CartPoleEvaluator(n, CartPoleEnvironment())
f= OppositeFunction(ee)
f.desiredValue = -500
    
for i in range(20):
    cl = CMAES(f, silent = False, maxEvals = 5000)#, x0 = n.getParameters())
    print ravel(cl.optimize()), ee.controlledExecute(n.getParameters())
    

    
    
    