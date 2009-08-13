# TODO, move to task.py ?

__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import power

from pybrain.utilities import abstractMethod
from pybrain.rl.environments.task import Task
from pybrain.rl.agents.agent import Agent
from pybrain.structure.modules.module import Module
# DEBUG: from pybrain.rl.evaluator import Evaluator
from pybrain.rl.experiments.episodic import EpisodicExperiment

class EpisodicTask(Task):  # DEBUG: also inherit from Evaluator
    """ A task that consists of independent episodes. """

    # tracking cumulative reward
    cumreward = 0
    
    # tracking the number of samples
    samples = 0
    
    # the discount factor 
    discount = None
    
    batchSize = 1
    
    def reset(self):
        """ reinitialize the environment """
        # Note: if a task needs to be reset at the start, the subclass constructor 
        # should take care of that.
        self.env.reset()
        self.cumreward = 0
        self.samples = 0        
        
    def isFinished(self): 
        """ is the current episode over? """
        abstractMethod()
        
    def performAction(self, action):
        Task.performAction(self, action)
        self.addReward()
        self.samples += 1
    
    def addReward(self):
        """ a filtered mapping towards performAction of the underlying environment. """                
        # by default, the cumulative reward is just the sum over the episode    
        if self.discount:
            self.cumreward += power(self.discount, self.samples) * self.getReward()
        else:
            self.cumreward += self.getReward()
    
    def getTotalReward(self):
        """ the accumulated reward since the start of the episode """
        return self.cumreward
        

        """ An episodic task can be used as an evaluation function of a module that produces actions 
        from observations, or as an evaluator of an agent. """
        if isinstance(module, Module):
            module.reset()
            self.reset()
            while not self.isFinished():
                self.performAction(module.activate(self.getObservation()))
            return self.getTotalReward()
        elif isinstance(module, Agent):
            EpisodicExperiment(self, module).doEpisodes(self.batchSize)
            return self.getTotalReward() / float(self.batchSize)
        else:
            raise NotImplementedError('Missing implementation for '+module.__class__.__name__+' evaluation')
        