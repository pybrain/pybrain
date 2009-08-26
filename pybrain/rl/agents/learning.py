__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.rl.agents.logging import LoggingAgent

class LearningAgent(LoggingAgent):
    """ LearningAgent has a module, a learner, that modifies the module, and an explorer,
        which perturbs the actions. It can have learning enabled or disabled and can be 
        used continously or with episodes.
    """
    
    def __init__(self, module, learner = None):
        """ 
        @param module: the acting module
        @param learner: the learner (optional) """
        
        LoggingAgent.__init__(self, module.indim, module.outdim)
        
        self.module = module
        self.learner = learner
                
        # if learner is available, tell it the module and data
        if self.learner:
            self.learner.module = self.module
            self.learner.dataset = self.history
        
        self.learning = True
        
        
    def _getLearning(self):
        """ returns whether the agent currently learns from experience or not. """
        return self.__learning
        
    
    def _setLearning(self, flag):
        """ set whether or not the agent should learn from its experience """
        if self.learner:
            self.__learning = flag
        else:
            self.__learning = False
        
    learning = property(_getLearning, _setLearning)
                
    
    def getAction(self):
        """ activates the module with the last observation, adds the exploration from
            the explorer object and stores the result as last action. """
        LoggingAgent.getAction(self)
        
        self.lastaction = self.module.activate(self.lastobs)
        
        if self.learning:
            self.lastaction = self.learner.explore(self.lastobs, self.lastaction)
            
        return self.lastaction
                    
    
    def reset(self):
        """ clears the history of the agent and resets the module. """
        LoggingAgent.reset(self)
        self.module.reset()
    
    
    def learn(self, episodes=1):
        """ calls the learner's learn function, which has access to both module and history. """
        if self.learning:
            self.learner.learnEpisodes(episodes)
    
