__author__ = "Thomas Rueckstiess, ruecksti@in.tum.de"

from scipy import random, array

from pybrain.rl.explorers.explorer import DiscreteExplorer


class EpsilonGreedyExplorer(DiscreteExplorer):
    """ A discrete explorer, that executes the original policy in most cases, 
        but sometimes returns a random action (uniformly drawn) instead. The
        randomness is controlled by a parameter 0 <= epsilon <= 1. The closer
        epsilon gets to 0, the more greedy (and less explorative) the agent
        behaves.
    """
    
    def __init__(self, epsilon = 0.5, decay = 0.9998):
        self.epsilon = epsilon
        self.decay = decay
    
    def activate(self, state, action):
        """ Draws a random number between 0 and 1. If the number is less
            than epsilon, a random action is chosen. If it is equal or
            larger than epsilon, the greedy action is returned.
        """
        assert self.module
        
        if random.random() < self.epsilon:
            result = array([random.randint(self.module.numColumns)])
        else:
            result = action
            
        self.epsilon *= self.decay
        
        return result
