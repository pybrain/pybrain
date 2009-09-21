__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.agents.agent import Agent

class OptimizationAgent(Agent):
    """ A simple wrapper to allow optimizers to conform to the RL interface.
        Works only in conjunction with EpisodicExperiment.
    """
    def __init__(self, module, learner):
        self.module = module
        self.learner = learner

