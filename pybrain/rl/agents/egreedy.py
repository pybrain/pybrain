from learning import LearningAgent
from scipy import random, array

class EpsilonGreedyAgent(LearningAgent):
    
    def __init__(self, module, learner):
        LearningAgent.__init__(self, module, learner)
        
        self.epsilon = 0.5
        self.epsilondecay = 0.9999
    
    def getAction(self):
        """ activates the module with the last observation and stores the result as last action. """
        # get greedy action
        action = LearningAgent.getAction(self)
        
        # explore by chance
        if random.random() < self.epsilon:
             action = array([random.randint(self.module.numActions)])
        
        # reduce epsilon
        self.epsilon *= self.epsilondecay
        
        return action
        

                                          