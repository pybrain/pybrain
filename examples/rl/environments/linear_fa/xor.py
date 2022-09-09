from __future__ import print_function

""" Toy example for RL with linear function approximation. 
This illustrates how a 'AND'-state-space can be solved, but not 
an 'XOR' space. 
"""

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.learners.valuebased.linearfa import Q_LinFA
from pybrain.rl.environments.classic.xor import XORTask
from pybrain.rl.experiments.experiment import Experiment
from pybrain.rl.agents.learning import LearningAgent
from random import random, randint

class LinFA_QAgent(LearningAgent):
    """ Customization of the Agent class for linear function approximation learners. """

    epsilon = 0.1
    logging = False
    
    def __init__(self, learner):
        self.learner = learner
        self.previousobs = None
        
    def getAction(self):
        if random() < self.epsilon:
            a = randint(0, self.learner.num_actions-1)
        else:
            a = self.learner._greedyAction(self.lastobs)  
        self.lastaction = a
        return a
    
    def giveReward(self, r):
        LearningAgent.giveReward(self, r)
        if self.previousobs is not None:
            #print  self.previousobs, a, self.lastreward, self.lastobs
            self.learner._updateWeights(self.previousobs, self.previousaction, self.previousreward, self.lastobs)
        self.previousobs = self.lastobs
        self.previousaction = self.lastaction
        self.previousreward = self.lastreward
        
    

def runExp(gamma=0, epsilon=0.1, xor=False, lr = 0.02):    
    if xor: 
        print("Attempting the XOR task")
    else:
        print("Attempting the AND task")
        
    task = XORTask()
    task.and_task = not xor
    
    l = Q_LinFA(task.nactions, task.nsenses)
    l.rewardDiscount = gamma
    l.learningRate = lr

    agent = LinFA_QAgent(l)
    agent.epsilon = epsilon
    exp = Experiment(task, agent)    
            
    sofar = 0
    for i in range(30):
        exp.doInteractions(100)
        print(exp.task.cumreward - sofar, end=' ')
        if i%10 == 9: 
            print()                
        sofar = exp.task.cumreward          
        l._decayLearningRate()


if __name__ == "__main__":
    runExp(xor=False)
    print() 
    runExp(xor=True)
    print() 
    runExp(xor=True)