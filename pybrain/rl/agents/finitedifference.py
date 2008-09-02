__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from learning import LearningAgent


class FiniteDifferenceAgent(LearningAgent):
    """ FiniteDifferenceAgent is a learning agent, that perturbs the parameters
        of the module directly and learns from the difference quotient of the
        resulting returns and the parameter deltas.
    """

    def enableLearning(self):
        """ if the agent can learn from experience, then this method enables learning. """
        self.learner.enableLearning()  
        self.learning = True
        
    def disableLearning(self):
        """ if the agent can learn from experience, then this method disables learning. """
        self.learner.disableLearning()  
        self.learning = False

    def setParameters(self, params):
        """ change the parameters of the module (wrapper function). """
        self.module._setParameters(params)

    def newEpisode(self):
        """ indicates a new episode in the training cycle. """
        LearningAgent.newEpisode(self)
        self.learner.perturbate()        
