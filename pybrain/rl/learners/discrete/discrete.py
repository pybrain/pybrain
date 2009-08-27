__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.rl.learners.learner import Learner
from pybrain.rl.explorers.discrete.egreedy import EpsilonGreedyExplorer

class DiscreteLearner(Learner):
    
    offPolicy = False
    batchMode = True

    _module = None
    _explorer = None

    def __init__(self):
        # create default explorer and hand it the module
        self.explorer = EpsilonGreedyExplorer()

    def _setModule(self, module):
        self.explorer.module = module
        self._module = module

    def _getModule(self):
        return self._module
        
    module = property(_getModule, _setModule)

    def _setExplorer(self, explorer):
        self._explorer = explorer
        if self.module:
            self._explorer.module = self.module

    def _getExplorer(self):
        return self._explorer
        
    explorer = property(_getExplorer, _setExplorer)
    
    