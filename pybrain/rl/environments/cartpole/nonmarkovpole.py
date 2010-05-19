__author__ = 'Tom Schaul, tom@idsia.ch'

from cartpole import CartPoleEnvironment


class NonMarkovPoleEnvironment(CartPoleEnvironment):
    """ CartPoleEnvironment which does not give access to the derivatives. """

    outdim = 2

    def getSensors(self):
        """ returns the state one step (dt) ahead in the future. stores the state in
            self.sensors because it is needed for the next calculation. The sensor return
            vector has 2 elements: theta, s
            (s being the distance from the origin).
        """
        tmp = CartPoleEnvironment.getSensors(self)
        return (tmp[0], tmp[2])

    def getCartPosition(self):
        """ auxiliary access to just the cart position, to be used by BalanceTask """
        return self.sensors[1]


