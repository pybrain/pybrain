from pybrain.rl.environments import Task
from scipy import array

class EMAMarketTask(Task):
    """ I don't know WTF this is
    """

    def getReward(self):
        """ compute and return the current reward (i.e. corresponding to the last action performed) """
        if self.env.goal == self.env.perseus:
            self.env.reset()
            reward = 1.
        else:
            reward = 0.
        return reward

    def performAction(self, action):
        """ The action vector is stripped and the only element is cast to integer and given
            to the super class.
        """
        Task.performAction(self, int(action[0]))


    def getObservation(self):
        """ The agent receives its position in the maze, to make this a fully observable
            MDP problem.
        """
        obs = array([self.env.perseus[0] * self.env.mazeTable.shape[0] + self.env.perseus[1]])
        return obs



