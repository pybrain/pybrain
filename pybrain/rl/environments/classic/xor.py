__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.environments.episodic import EpisodicTask
from scipy import array
from random import randint, random


class XORTask(EpisodicTask):
    """ Continuous task, producing binary observations, taking a single, binary action
    rewarding the agent whenever action = xor(obs).
    """
    
    nactions = 2
    nsenses = 3
    
    randomorder = False
    
    and_task = False
    stochasticity = 0
    
    def __init__(self):
        self.r = 0
        self._counter = 0
        
    def getObservation(self):
        if self.randomorder:
            self.obs = array([randint(0,1), randint(0,1), 1])
        else:
            self.obs = array([self._counter%2, (self._counter/2)%2, 1])    
        self._counter += 1    
        return self.obs
        
    def performAction(self, action):
        if ((self.and_task and (action == self.obs[0] & self.obs[1]))
            or (not self.and_task and action == self.obs[0] ^ self.obs[1])):
            self.r = 1
        else:
            self.r = -1
        #print(self.obs, action, self.r    )
        self.addReward()       
            
    def getReward(self):
        if random() < self.stochasticity:
            return -self.r
        else:
            return self.r
    
    def isFinished(self):
        return False
    
    
    
class XORChainTask(XORTask):
    """ Continuous task, producing binary observations, taking a single, binary action
    rewarding the agent whenever action = xor(obs).
    It is a chain, going back to the initial state whenever the bad action is taken.
    Reward increases as we move along the chain.
    """
    
    reward_cutoff = 0
    
    def __init__(self):
        self.r = 0
        self.state = 0
        
    def getObservation(self):
        self.obs = array([self.state%2, (self.state/2)%2, 1])    
        return self.obs
        
    def performAction(self, action):
        if ((self.and_task and action == self.obs[0] & self.obs[1])
            or (not self.and_task and action == self.obs[0] ^ self.obs[1])):
            self.r = -1+2*(self.state >= self.reward_cutoff)
            self.state = min(self.state+1, 3)
        else:
            self.r = -1
            self.state = 0   
        self.addReward()       
            