__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.utilities import abstractMethod
from task import Task

class EpisodicTask(Task):
    """ A task that consists of independent episodes. """

    # tracking cumulative reward
    cumreward = 0
    
    def reset(self):
        """ reinitialize the environment """
        # Note: if a task needs to be reset at the start, the subclass constructor 
        # should take care of that.
        self.env.reset()
        self.cumreward = 0        
        
    def isFinished(self): 
        """ is the current episode over? """
        abstractMethod()
        
    def performAction(self, action):
        Task.performAction(self, action)
        self.addReward()
    
    def addReward(self):
        """ a filtered mapping towards performAction of the underlying environment. """                
        # by default, the cumulative reward is just the sum over the episode    
        self.cumreward += self.getReward()
    
    def getTotalReward(self):
        """ the accumulated reward since the start of the episode """
        return self.cumreward
        
    def evaluateModule(self, module, averageOver = 1):
        """ Evaluate the interactions of a module with a task for one episode
        and return the total reward.
        (potentially average over a number of episodes). """
        res = 0.
        for dummy in range(averageOver):
            module.reset()
            self.reset()
            while not self.isFinished():
                self.performAction(module.activate(self.getObservation()))
            res += self.getTotalReward()
        return res/averageOver