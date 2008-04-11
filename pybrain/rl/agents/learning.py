__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from history import HistoryAgent

class LearningAgent(HistoryAgent):
    """ LearningAgent has a module and a learner, that modifies the module. It can
        have learning enabled or disabled and can be used continously or with episodes.
    """
    
    def __init__(self, module, learner = None):
        """ 
        @param module: the acting module
        @param learner: the learner (optional) """
        HistoryAgent.__init__(self, module.indim, module.outdim)
        self.module = module
        self.learner = learner
        if self.learner:
            self.learner.setModule(self.module)
            self.learner.setData(self.history)
            self.learning = True
        else:
            self.learning = False
        
    def enableLearning(self):
        """ if the agent can learn from experience, then this method enables learning. """
        if self.learner:
            self.learning = True
        
    def disableLearning(self):
        """ if the agent can learn from experience, then this method disables learning. """
        self.learning = False
        
    def getAction(self):
        """ activates the module with the last observation and stores the result as last action. """
        HistoryAgent.getAction(self)
        self.lastaction = self.module.activate(self.lastobs)
        return self.lastaction
        
    def newEpisode(self):
        self.history.newSequence()  
            
    def reset(self):
        """ clears the history of the agent and resets the module. """
        HistoryAgent.reset(self)
        self.module.reset()
    
    def learn(self, epochs=1):
        """ calls the learner's learn function, which has access to both module and history. """
        if self.learning:
            self.learner.learnEpochs(epochs)
    
