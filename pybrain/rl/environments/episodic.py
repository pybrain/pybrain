__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import power

from pybrain.utilities import abstractMethod
from pybrain.rl.environments.task import Task
from pybrain.rl.agents.agent import Agent
from pybrain.structure.modules.module import Module
from pybrain.rl.environments.fitnessevaluator import FitnessEvaluator
from pybrain.rl.experiments.episodic import EpisodicExperiment


class EpisodicTask(Task, FitnessEvaluator):
    """ A task that consists of independent episodes. """

    # tracking cumulative reward
    cumreward = 0

    # tracking the number of samples
    samples = 0

    #: Discount factor
    discount = None

    batchSize = 1

    def reset(self):
        """ Re-initialize the environment """
        # Note: if a task needs to be reset at the start, the subclass constructor
        # should take care of that.
        self.env.reset()
        self.cumreward = 0
        self.samples = 0

    def isFinished(self):
        """ Is the current episode over? """
        abstractMethod()

    def performAction(self, action):
        """ Execute one action. """
        Task.performAction(self, action)
        self.samples += 1
        self.addReward()
        
    def addReward(self):
        """ A filtered mapping towards performAction of the underlying environment. """
        # by default, the cumulative reward is just the sum over the episode
        if self.discount:
            self.cumreward += power(self.discount, self.samples) * self.getReward()
        else:
            self.cumreward += self.getReward()

    def getTotalReward(self):
        """ Return the accumulated reward since the start of the episode """
        return self.cumreward

    def f(self, x):
        """ An episodic task can be used as an evaluation function of a module that produces actions
        from observations, or as an evaluator of an agent. """
        r = 0.
        for _ in range(self.batchSize):
            if isinstance(x, Module):
                x.reset()
                self.reset()
                while not self.isFinished():
                    self.performAction(x.activate(self.getObservation()))
            elif isinstance(x, Agent):
                EpisodicExperiment(self, x).doEpisodes()
            else:
                raise ValueError(self.__class__.__name__+' cannot evaluate the fitness of '+str(type(x)))
            r += self.getTotalReward()
        return r / float(self.batchSize)
