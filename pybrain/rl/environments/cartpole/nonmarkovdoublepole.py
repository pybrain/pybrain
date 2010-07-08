__author__ = 'Tom Schaul, tom@idsia.ch'

from doublepole import DoublePoleEnvironment
from nonmarkovpole import NonMarkovPoleEnvironment


class NonMarkovDoublePoleEnvironment(DoublePoleEnvironment, NonMarkovPoleEnvironment):
    """ DoublePoleEnvironment which does not give access to the derivatives. """

    outdim = 3

    def getSensors(self):
        """ returns the state one step (dt) ahead in the future. stores the state in
            self.sensors because it is needed for the next calculation. The sensor return
            vector has 3 elements: theta1, theta2, s
            (s being the distance from the origin).
        """
        tmp = DoublePoleEnvironment.getSensors(self)
        return (tmp[0], tmp[2], tmp[4])

    def getPoleAngles(self):
        """ auxiliary access to just the pole angle(s), to be used by BalanceTask """
        return [self.sensors[0], self.sensors[1]]

    def getCartPosition(self):
        """ auxiliary access to just the cart position, to be used by BalanceTask """
        return self.sensors[2]


