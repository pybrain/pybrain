__author__ = 'Tom Schaul, tom@idsia.ch'


class OptimizationAgent(object):
    """ A simple wrapper to allow optimizers to conform to the RL interface.
    Works only in conjunction with EpisodicExperiment.
    """
    def __init__(self, module, learner):
        self.module = module
        self.learner = learner

