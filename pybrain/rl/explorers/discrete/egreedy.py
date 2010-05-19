__author__ = "Thomas Rueckstiess, ruecksti@in.tum.de"

from scipy import random, array

from pybrain.rl.explorers.discrete.discrete import DiscreteExplorer

class EpsilonGreedyExplorer(DiscreteExplorer):
    """ A discrete explorer, that executes the original policy in most cases,
        but sometimes returns a random action (uniformly drawn) instead. The
        randomness is controlled by a parameter 0 <= epsilon <= 1. The closer
        epsilon gets to 0, the more greedy (and less explorative) the agent
        behaves.
    """

    def __init__(self, epsilon = 0.3, decay = 0.9999):
        DiscreteExplorer.__init__(self)
        self.epsilon = epsilon
        self.decay = decay

    def _forwardImplementation(self, inbuf, outbuf):
        """ Draws a random number between 0 and 1. If the number is less
            than epsilon, a random action is chosen. If it is equal or
            larger than epsilon, the greedy action is returned.
        """
        assert self.module

        if random.random() < self.epsilon:
            outbuf[:] = array([random.randint(self.module.numActions)])
        else:
            outbuf[:] = inbuf

        self.epsilon *= self.decay


