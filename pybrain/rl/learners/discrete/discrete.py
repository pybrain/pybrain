__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.rl.learners.learner import Learner
from pybrain.rl.explorers.discrete.egreedy import EpsilonGreedyExplorer

class DiscreteLearner(Learner):
    
    offPolicy = False
    batchMode = True
    
    explorer = EpsilonGreedyExplorer()
    