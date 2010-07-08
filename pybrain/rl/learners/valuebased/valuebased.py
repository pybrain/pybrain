__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.rl.learners.learner import ExploringLearner, DataSetLearner, EpisodicLearner
from pybrain.rl.explorers.discrete.egreedy import EpsilonGreedyExplorer


class ValueBasedLearner(ExploringLearner, DataSetLearner, EpisodicLearner):
    """ An RL algorithm based on estimating a value-function."""

    #: Does the algorithm work on-policy or off-policy?
    offPolicy = False

    #: Does the algorithm run in batch mode or online?
    batchMode = True

    _module = None
    _explorer = None

    def __init__(self):
        """ Create a default explorer for discrete learning tasks. """
        self.explorer = EpsilonGreedyExplorer()

    def _setModule(self, module):
        """ Set module and tell explorer about the module. """
        if self.explorer:
            self.explorer.module = module
        self._module = module

    def _getModule(self):
        """ Return the internal module. """
        return self._module

    module = property(_getModule, _setModule)

    def _setExplorer(self, explorer):
        """ Set explorer and tell it the module, if already available. """
        self._explorer = explorer
        if self.module:
            self._explorer.module = self.module

    def _getExplorer(self):
        """ Return the internal explorer. """
        return self._explorer

    explorer = property(_getExplorer, _setExplorer)


