__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.rl.agents.logging import LoggingAgent


class LearningAgent(LoggingAgent):
    """ LearningAgent has a module, a learner, that modifies the module, and an explorer,
        which perturbs the actions. It can have learning enabled or disabled and can be
        used continuously or with episodes.
    """

    def __init__(self, module, learner = None):
        """
        :key module: the acting module
        :key learner: the learner (optional) """

        LoggingAgent.__init__(self, module.indim, module.outdim)

        self.module = module
        self.learner = learner

        # if learner is available, tell it the module and data
        if self.learner is not None:
            self.learner.module = self.module
            self.learner.dataset = self.history

        self.learning = True


    def _getLearning(self):
        """ Return whether the agent currently learns from experience or not. """
        return self.__learning


    def _setLearning(self, flag):
        """ Set whether or not the agent should learn from its experience """
        if self.learner is not None:
            self.__learning = flag
        else:
            self.__learning = False

    learning = property(_getLearning, _setLearning)


    def getAction(self):
        """ Activate the module with the last observation, add the exploration from
            the explorer object and store the result as last action. """
        LoggingAgent.getAction(self)

        self.lastaction = self.module.activate(self.lastobs)

        if self.learning:
            self.lastaction = self.learner.explore(self.lastobs, self.lastaction)

        return self.lastaction


    def newEpisode(self):
        """ Indicate the beginning of a new episode in the training cycle. """
        # reset the module when a new episode starts.
        self.module.reset()
        
        if self.logging:
            self.history.newSequence()

        # inform learner about the start of a new episode
        if self.learning:
            self.learner.newEpisode()

    def reset(self):
        """ Clear the history of the agent and resets the module and learner. """
        LoggingAgent.reset(self)
        self.module.reset()
        if self.learning:
            self.learner.reset()


    def learn(self, episodes=1):
        """ Call the learner's learn method, which has access to both module and history. """
        if self.learning:
            self.learner.learnEpisodes(episodes)

