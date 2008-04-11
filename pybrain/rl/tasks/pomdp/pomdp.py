__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.tasks import EpisodicTask
from pybrain.utilities import Named


class POMDPTask(EpisodicTask, Named):
    """ Partially observable episodic MDP (with discrete actions)
    Has actions that can be performed, and observations in every state.
    By default, the observation is a vector, and the actions are integers.
    """
    # number of observations
    observations = 4
    
    # number of possible actions
    actions = 4
    
    # the discount factor 
    discount = None
    
    # maximal number of steps before the episode is stopped
    maxSteps = None
    
    # the lower bound on the reward value
    minReward = 0
    
    def __init__(self, **args):
        self.setArgs(**args)
        self.steps = 0
        
    def getInDim(self):
        return self.actions
    
    def getOutDim(self):
        return self.observations
    
    def reset(self):
        self.steps = 0
        EpisodicTask.reset(self)
        
    def isFinished(self):
        if self.maxSteps != None:
            return self.steps >= self.maxSteps
        return False
    
    def performAction(self, action):
        self.steps += 1
        EpisodicTask.performAction(self, action)