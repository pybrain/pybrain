from pybrain.rl.learners.rllearner import RLLearner

class QLearner(RLLearner):
    
    alpha = 0.1
    gamma = 0.5
    
    def learn(self):
        """ learn on the current dataset, for a single step. """
        state, action, reward = self.ds.getSample()
        state = int(state)
        action = int(action)
        
        qvalue = self.module.getValue(state, action)
        maxvalue = self.module.getValue(state, self.module.getMaxAction(state))
        self.module.updateValue(state, action, qvalue + self.alpha * (reward + self.gamma * maxvalue - qvalue))
