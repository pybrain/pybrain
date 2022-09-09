__author__ = "Thomas Rueckstiess, ruecksti@in.tum.de"


from pybrain.rl.explorers.discrete.discrete import DiscreteExplorer
from pybrain.rl.learners.valuebased.interface import ActionValueTable, ActionValueNetwork
from copy import deepcopy
from numpy import random, array


class DiscreteStateDependentExplorer(DiscreteExplorer):
    """ A discrete explorer, that directly manipulates the ActionValue
        estimator (table or network) and keeps the changes fixed for one
        full episode (if episodic) or slowly changes it over time.

        TODO: currently only implemented for episodes
    """

    def __init__(self, epsilon = 0.2, decay = 0.9998):
        """ TODO: the epsilon and decay parameters are currently
            not implemented.
        """
        DiscreteExplorer.__init__(self)
        self.state = None

    def _setModule(self, module):
        """ Tell the explorer the module. """
        self._module = module
        # copy the original module for exploration
        self.explorerModule = deepcopy(module)

    def _getModule(self):
        return self._module

    module = property(_getModule, _setModule)


    def activate(self, state, action):
        """ Save the current state for state-dependent exploration. """
        self.state = state
        return DiscreteExplorer.activate(self, state, action)

    def _forwardImplementation(self, inbuf, outbuf):
        """ Activate the copied module instead of the original and
            feed it with the current state.
        """
        if random.random() < 0.001:
            outbuf[:] = array([random.randint(self.module.numActions)])
        else:
            outbuf[:] = self.explorerModule.activate(self.state)

    def newEpisode(self):
        """ Inform the explorer about the start of a new episode. """
        self.explorerModule = deepcopy(self.module)

        if isinstance(self.explorerModule, ActionValueNetwork):

            self.explorerModule.network.mutationStd = 0.01
            self.explorerModule.network.mutate()

        elif isinstance(self.explorerModule, ActionValueTable):
            self.explorerModule.mutationStd = 0.01
            self.explorerModule.mutate()
