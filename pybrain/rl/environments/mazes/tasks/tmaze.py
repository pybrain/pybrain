__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import array, zeros
from random import choice

from maze import MazeTask


class TMaze(MazeTask):
    """
    #############
    ###########*#
    #.          #
    ########### #
    #############

    1-in-n encoding for observations.
    """

    discount = 0.98
    observations = 4

    finalReward = 4
    bangPenalty = -0.1

    length = 10

    def __init__(self, **args):
        self.initPos = [(2, 1)]
        self.setArgs(**args)
        columns = [[1] * 5]
        for dummy in range(self.length):
            columns.append([1, 1, 0, 1, 1])
        columns.append([1, 0, 0, 0, 1])
        columns.append([1] * 5)
        self.topology = array(columns).T
        MazeTask.__init__(self, **args)

    def reset(self):
        MazeTask.reset(self)
        goUp = choice([True, False])
        self.specialObs = goUp
        if goUp:
            self.env.goal = (3, self.length + 1)
        else:
            self.env.goal = (1, self.length + 1)

    def getObservation(self):
        res = zeros(4)
        if self.env.perseus == self.env.initPos[0]:
            if self.specialObs:
                res[0] = 1
            else:
                res[1] = 1
        elif self.env.perseus[1] == self.length + 1:
            res[2] = 1
        else:
            res[3] = 1
        return res

    def getReward(self):
        if self.env.perseus[1] == self.length + 1:
            if abs(self.env.perseus[0] - self.env.goal[0]) == 2:
                # bad choice taken
                self.env.perseus = self.env.goal
                return self.bangPenalty
        return MazeTask.getReward(self)
