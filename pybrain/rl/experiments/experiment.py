__author__ = 'Tom Schaul, tom@idsia.ch'


class Experiment(object):
    """ An experiment matches up a task with an agent and handles their interactions.
    """

    def __init__(self, task, agent):
        self.task = task
        self.agent = agent
        self.stepid = 0

    def doInteractions(self, number = 1):
        """ The default implementation directly maps the methods of the agent and the task.
            Returns the number of interactions done.
        """
        for dummy in range(number):
            reward = self._oneInteraction()
        return self.stepid

    def _oneInteraction(self):
        """ gives the observation to the agent, takes its resulting action and returns
            it to the task. then gives the reward to the agent again and returns it.
        """
        self.stepid += 1
        self.agent.integrateObservation(self.task.getObservation())
        self.task.performAction(self.agent.getAction())
        reward = self.task.getReward()
        self.agent.giveReward(reward)
        return reward
