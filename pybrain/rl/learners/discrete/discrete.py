__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.rl.learners.learner import Learner
from pybrain.rl.explorers.discrete.egreedy import EpsilonGreedyExplorer

class DiscreteLearner(Learner):
    
    offPolicy = False
    batchMode = True

    __module = None

    def __init__(self):
        # create default explorer and hand it the module
        self.explorer = EpsilonGreedyExplorer()

    def _setModule(self, module):
        self.explorer.module = module
        self.__module = module

    def _getModule(self):
        return self.__module

    module = property(_getModule, _setModule)