__author__ = 'Tom Schaul, tom@idsia.ch'

from random import random, choice
from scipy import array, zeros

from maze import MazeTask


class FourByThreeMaze(MazeTask):
    """
    ######
    #   *#
    # # -#
    #    #
    ######

    The '-' spot if absorbing, and giving negative reward.
    """

    discount = 0.95

    defaultPenalty = -0.04
    bangPenalty = -0.04
    minReward = -1

    stochAction = 0.1

    initPos = [(1, 1), (1, 2), (1, 3), (2, 1), (3, 1), (3, 2), (2, 3), (3, 3), (1, 4)]
    topology = array([[1] * 6,
                      [1, 0, 0, 0, 0, 1],
                      [1, 0, 1, 0, 0, 1],
                      [1, 0, 0, 0, 0, 1],
                      [1] * 6, ])
    goal = (3, 4)

    def reset(self):
        MazeTask.reset(self)
        self.bad = False

    def performAction(self, action):
        poss = []
        for a in range(self.actions):
            if action - a % 4 != 2:
                poss.append(a)
        if random() < self.stochAction * len(poss):
            MazeTask.performAction(self, choice(poss))
        else:
            MazeTask.performAction(self, action)

    def getReward(self):
        if self.bad:
            return self.minReward
        else:
            return MazeTask.getReward(self)

    def getObservation(self):
        """only walls on w, E, both or neither are observed. """
        res = zeros(4)
        all = self.env.getSensors()
        res[0] = all[3]
        res[1] = all[1]
        res[2] = all[3] and all[1]
        res[3] = not all[3] and not all[1]
        return res
