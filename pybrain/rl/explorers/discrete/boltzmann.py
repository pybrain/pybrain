__author__ = "Thomas Rueckstiess, ruecksti@in.tum.de"

from scipy import array

from pybrain.rl.explorers.discrete.discrete import DiscreteExplorer
from pybrain.utilities import drawGibbs

class BoltzmannExplorer(DiscreteExplorer):
    """ A discrete explorer, that executes the actions with probability
        that depends on their action values. The boltzmann explorer has
        a parameter tau (the temperature). for high tau, the actions are
        nearly equiprobable. for tau close to 0, this action selection
        becomes greedy.
    """

    def __init__(self, tau = 2., decay = 0.9995):
        DiscreteExplorer.__init__(self)
        self.tau = tau
        self.decay = decay
        self._state = None

    def activate(self, state, action):
        """ The super class ignores the state and simply passes the
            action through the module. implement _forwardImplementation()
            in subclasses.
        """
        self._state = state
        return DiscreteExplorer.activate(self, state, action)


    def _forwardImplementation(self, inbuf, outbuf):
        """ Draws a random number between 0 and 1. If the number is less
            than epsilon, a random action is chosen. If it is equal or
            larger than epsilon, the greedy action is returned.
        """
        assert self.module

        values = self.module.getActionValues(self._state)
        action = drawGibbs(values, self.tau)

        self.tau *= self.decay

        outbuf[:] = array([action])
