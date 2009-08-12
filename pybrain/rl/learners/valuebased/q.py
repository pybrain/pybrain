__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.rl.learners.learner import Learner
from pybrain.rl.explorers.discrete.egreedy import EpsilonGreedyExplorer


class Q(Learner):
    
    offPolicy = True
    defaultExploration = EpsilonGreedyExplorer
    
    
    def __init__(self):
        self.alpha = 0.5
        self.gamma = 0.99
    
        self.laststate = None
        self.lastaction = None
    
    def learn(self):
        """ learn on the current dataset, either for many timesteps and
            even episodes (batch mode) or for a single timestep, if the
            dataset only contains one sample. Batch mode is possible,
            because Q-Learning is an off-policy method.
        """
        
        # go through all sequences and timesteps and apply the q-learning
        # updates iteratively. in continuous mode, this will actually be
        # called after each timestep and only update one single value.
        for seq in self.dataset:
            for state, action, reward in seq:
                
                state = int(state)
                action = int(action)
        
                # first learning call has no last state: skip
                if self.laststate == None:
                    self.lastaction = action
                    self.laststate = state
                    continue
        
                qvalue = self.module.getValue(self.laststate, self.lastaction)
                maxnext = self.module.getValue(state, self.module.getMaxAction(state))
                self.module.updateValue(self.laststate, self.lastaction, qvalue + self.alpha * (reward + self.gamma * maxnext - qvalue))
        
                # move state to oldstate
                self.laststate = state
                self.lastaction = action
