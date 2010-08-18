__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.agents.logging import LoggingAgent
from pybrain.utilities import drawIndex


class LinearFA_Agent(LoggingAgent):
    """ Agent class for using linear-FA RL algorithms. """    
    
    exploration_decay = 0.9995
    init_epsilon = 0.1
    init_temperature = 1.
    
    greedy = False
    # default is boltzmann exploration 
    epsilonGreedy = False
       
    learning = True
     
    def __init__(self, learner, **kwargs):
        LoggingAgent.__init__(self, learner.num_features, 1, **kwargs)
        self.learner = learner
        self.learner._behaviorPolicy = self._actionProbs
        self.reset()
        
    def _actionProbs(self, state):
        if self.greedy:
            return self.learner._greedyPolicy(state)
        elif self.epsilonGreedy:
            return (self.learner._greedyPolicy(state) * (1-self._ra_proportion) 
                    + self._ra_proportion/float(self.learner.num_actions))
        else:
            return self.learner._boltzmannPolicy(state, self._temperature)            
        
    def getAction(self):
        self.lastaction = drawIndex(self._actionProbs(self.lastobs), True)
        self._temperature *= self.exploration_decay
        self._ra_proportion *= self.exploration_decay
        return [self.lastaction]
        
    def reset(self):
        LoggingAgent.reset(self)
        self._temperature = self.init_temperature
        self._ra_proportion = self.init_epsilon
        self.learner.reset()    
        self.newEpisode()
        
    def newEpisode(self):
        """ Indicate the beginning of a new episode in the training cycle. """
        if self.logging:
            self.history.newSequence()
        if self.learning:
            self.learner.newEpisode()
            
    def learn(self):
        if not self.learning:
            return
        for seq in self.history:
            for obs, action, reward in seq:
                if self.laststate is not None:
                    self.learner._updateWeights(self.lastobs, self.lastaction, self.lastreward, obs)
                self.lastobs = obs
                self.lastaction = action[0]
                self.lastreward = reward
            self.learner.newEpisode()
