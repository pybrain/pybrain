__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.rl.learners.learner import ExploringLearner, DataSetLearner, OntogeneticLearner
from pybrain.rl.explorers.discrete.egreedy import EpsilonGreedyExplorer


class ValueBasedLearner(OntogeneticLearner, ExploringLearner, DataSetLearner):
    
    offPolicy = False
    batchMode = True

    _module = None
    _explorer = None

    def __init__(self):
        """ create a default explorer for discrete learning tasks. """
        self.explorer = EpsilonGreedyExplorer()

    def _setModule(self, module):
        """ set module and tell explorer about the module. """
        self.explorer.module = module
        self._module = module

    def _getModule(self):
        """ return the module. """
        return self._module
        
    module = property(_getModule, _setModule)

    def _setExplorer(self, explorer):
        """ set explorer and tell it the module, if already available. """
        self._explorer = explorer
        if self.module:
            self._explorer.module = self.module

    def _getExplorer(self):
        """ return the explorer. """
        return self._explorer
        
    explorer = property(_getExplorer, _setExplorer)
    
    
