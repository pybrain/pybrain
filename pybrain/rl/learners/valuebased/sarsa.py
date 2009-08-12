__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.rl.learners.learner import Learner
from pybrain.rl.explorers.discrete.egreedy import EpsilonGreedyExplorer


class SARSA(Learner):
    
    offPolicy = False
    defaultExploration = EpsilonGreedyExplorer
    
    def __init__(self):
        self.alpha = 0.5
        self.gamma = 0.99
    
        self.laststate = None
        self.lastaction = None

    def learn(self):
        """ learn on the current dataset, for a single step. This algorithm is
            on-policy and can thus not perform batch updates. """
        
        state, action, reward = self.dataset.getSample()
                
        state = int(state)
        action = int(action)
        
        # first learning call has no last state: skip
        if self.laststate == None:
            self.lastaction = action
            self.laststate = state
            return
        
        qvalue = self.module.getValue(self.laststate, self.lastaction)
        qnext = self.module.getValue(state, action)
        self.module.updateValue(self.laststate, self.lastaction, qvalue + self.alpha * (reward + self.gamma * qnext - qvalue))
        
        # move state to oldstate
        self.laststate = state
        self.lastaction = action
