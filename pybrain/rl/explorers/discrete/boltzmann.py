__author__ = "Thomas Rueckstiess, ruecksti@in.tum.de"

from scipy import random, array

from pybrain.rl.explorers.explorer import DiscreteExplorer
from pybrain.utilities import drawGibbs

class BoltzmannExplorer(DiscreteExplorer):
    """ A discrete explorer, that executes the actions with probability 
        that depends on their action values. The boltzmann explorer has 
        a parameter tau (the temperature). for high tau, the actions are 
        nearly equiprobable. for tau close to 0, this action selection
        becomes greedy.
    """
    
    def __init__(self, tau = 5., decay = 0.99):
        self.tau = tau
        self.decay = decay
    
    def activate(self, state, action):
        """ Draws a random number between 0 and 1. If the number is less
            than epsilon, a random action is chosen. If it is equal or
            larger than epsilon, the greedy action is returned.
        """
        assert self.module 
        
        values = self.module.values[state, :].flatten()
        action = drawGibbs(values, self.tau)
        
        self.tau *= self.decay
        
        return array([action])
