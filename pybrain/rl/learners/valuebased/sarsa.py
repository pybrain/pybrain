__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.rl.learners.valuebased.valuebasedlearner import ValueBasedLearner
from pybrain.rl.learners.datasetlearner import DataSetLearner


class SARSA(ValueBasedLearner, DataSetLearner):
    
    def __init__(self, nActions):
        self.alpha = 0.5
        self.gamma = 0.99
    
        self.laststate = None
        self.lastaction = None
        
        self.nActions = nActions

    def learn(self):
        """ learn on the current dataset, for a single step. """
        """ TODO: also learn on episodic tasks (sum over whole sequence) """
        state, action, reward = self.ds.getSample()
                
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