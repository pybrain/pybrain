__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import zeros, array

from maze import MazeTask


class CheeseMaze(MazeTask):
    """
    #######
    #     #
    # # # #
    # #*# #
    #######
    """

    observations = 7
    discount = 0.95

    topology = array([[1] * 7,
                      [1, 0, 1, 0, 1, 0, 1],
                      [1, 0, 1, 0, 1, 0, 1],
                      [1, 0, 0, 0, 0, 0, 1],
                      [1] * 7])
    goal = (1, 3)

    def getObservation(self):
        """ observations are encoded in a 1-n encoding of possible wall combinations. """
        res = zeros(7)
        obs = self.env.getSensors()
        if self.env.perseus == self.env.goal:
            res[6] = 1
        elif sum(obs) == 3:
            res[0] = 1
        elif sum(obs) == 1:
            res[5] = 1
        elif obs[0] == obs[1]:
            res[1] = 1
        elif obs[0] == obs[3]:
            res[2] = 1
        elif obs[0] == obs[2]:
            if obs[0] == 1:
                res[3] = 1
            else:
                res[4] = 1
        return res
