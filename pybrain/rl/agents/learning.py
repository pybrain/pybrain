__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.rl.agents.logging import LoggingAgent
from pybrain.rl.explorers.explorer import DiscreteExplorer
from pybrain.rl.learners.continuous.policygradients.policygradient import PolicyGradientLearner

class LearningAgent(LoggingAgent):
    """ LearningAgent has a module, a learner, that modifies the module, and an explorer,
        which perturbs the actions. It can have learning enabled or disabled and can be 
        used continously or with episodes.
    """
    
    lastexploration = None
    
    def __init__(self, module, learner = None, explorer = None):
        """ 
        @param module: the acting module
        @param learner: the learner (optional) """
        
        LoggingAgent.__init__(self, module.indim, module.outdim)
        
        self.module = module
        self.learner = learner
        self.explorer = explorer
        
        # policy gradients need another type of dataset
        if isinstance(self.module, PolicyGradientLearner):
            self.history = PolicyGradientDataSet(self.indim, self.outdim)
        
        # if learner is available, tell it the module and data
        if self.learner:
            self.learner.setModule(self.module)
            self.learner.setData(self.history)
        
        # if no explorer is given, try the learning algorithm's default explorer
        if not self.explorer:    
            try:
                self.explorer = self.learner.defaultExploration()
            except(Exception):
                self.explorer = None            
        
        if self.explorer:
            # discrete explorers need access to the module
            if isinstance(self.explorer, DiscreteExplorer):
                self.explorer.setModule(self.module)

            
        self.learning = True
        
    def _getLearning(self):
        """ returns whether the agent currently learns from experience or not. """
        return self.__learning
        
    def _setLearning(self, flag):
        """ set whether or not the agent should learn from its experience """
        if self.learner and self.explorer:
            self.__learning = flag
        else:
            self.__learning = False
        
    learning = property(_getLearning, _setLearning)
    
    
    def _storeExperience(self):
        if isinstance(self.learner, PolicyGradientLearner):
            # policy gradients need state, action, exploration and reward in dataset
            self.history.addSample(self.lastobs, self.lastaction, self.lastexploration, self.lastreward)
        else:
            # store state, action and reward in dataset
            self.history.addSample(self.lastobs, self.lastaction, self.lastreward)
            
    
    def getAction(self):
        """ activates the module with the last observation, adds the exploration from
            the explorer object and stores the result as last action. """
        LoggingAgent.getAction(self)
        
        self.lastaction = self.module.activate(self.lastobs)
        self.lastexploration = self.lastaction
        
        if self.learning:
            self.lastexploration = self.explorer.activate(self.lastobs, self.lastaction)
        else:
            self.lastexploration = self.lastaction
            
        return self.lastexploration
                    
    
    def reset(self):
        """ clears the history of the agent and resets the module. """
        LoggingAgent.reset(self)
        self.module.reset()
    
    
    def learn(self, episodes=1):
        """ calls the learner's learn function, which has access to both module and history. """
        if self.learning:
            self.learner.learnEpisodes(episodes)
    
