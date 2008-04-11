__author__ = 'Tom Schaul, tom@idsia.ch'

from episodicevaluator import EpisodicEvaluator
from pybrain.rl.environments.cartpole import BalanceTask

    
class CartPoleEvaluator(EpisodicEvaluator):
    """ The fitness function is given by the number of timesteps the controlling module can 
    balance a pole on a cart. """
            
    def __init__(self, module, env = None):
        EpisodicEvaluator.__init__(self, module, BalanceTask(env = env))
        self.desiredValue = self.task.N - 1
        # a simpler fitness: total number of balanced steps
        self.task.getTotalReward = lambda: self.task.t
        
        