__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.rl.learners.learner import Learner
from pybrain.rl.explorers.egreedy import EpsilonGreedyExplorer


class QLambda(Learner):
    
    offPolicy = True
    defaultExploration = EpsilonGreedyExplorer
    
    
    def __init__(self):
        self.alpha = 0.5
        self.gamma = 0.99
        self.qlambda = 0.9
        
        self.laststate = None
        self.lastaction = None
        
    
    def learn(self):
        """ learn on the current dataset with an eligibility trace. """
        
        states = self.ds['state']
        actions = self.ds['action']
        rewards = self.ds['reward']
        
        for i in range(states.shape[0]-1, 0, -1):
            lbda = self.qlambda**(states.shape[0]-1-i)
            # if eligibility trace gets too long, break
            if lbda < 0.0001:
                break
                
            state = int(states[i])
            laststate = int(states[i-1])
            action = int(actions[i])
            lastaction = int(actions[i-1])
            reward = int(rewards[i])

        
            qvalue = self.module.getValue(laststate, lastaction)
            maxnext = self.module.getValue(state, self.module.getMaxAction(state))
            self.module.updateValue(laststate, lastaction, qvalue + self.alpha * lbda * (reward + self.gamma * maxnext - qvalue))
