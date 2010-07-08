__author__ = 'Tom Schaul, tom@idsia.ch'

from cartpole import CartPoleEnvironment
from pybrain.rl.environments import Environment


class DoublePoleEnvironment(Environment):
    """ two poles to be balanced from the same cart. """

    indim = 1
    ooutdim = 6

    def __init__(self):
        self.p1 = CartPoleEnvironment()
        self.p2 = CartPoleEnvironment()
        self.p2.l = 0.05
        self.p2.mp = 0.01
        self.reset()

    def getSensors(self):
        """ returns the state one step (dt) ahead in the future. stores the state in
            self.sensors because it is needed for the next calculation. The sensor return
            vector has 6 elements: theta1, theta1', theta2, theta2', s, s'
            (s being the distance from the origin).
        """
        s1 = self.p1.getSensors()
        s2 = self.p2.getSensors()
        self.sensors = (s1[0], s1[1], s2[0], s2[1], s2[2], s2[3])
        return self.sensors

    def reset(self):
        """ re-initializes the environment, setting the cart back in a random position.
        """
        self.p1.reset()
        self.p2.reset()
        # put cart in the same place:
        self.p2.sensors = (self.p2.sensors[0], self.p2.sensors[1], self.p1.sensors[2], self.p1.sensors[3])
        self.getSensors()

    def performAction(self, action):
        """ stores the desired action for the next runge-kutta step.
        """
        self.p1.performAction(action)
        self.p2.performAction(action)

    def getCartPosition(self):
        """ auxiliary access to just the cart position, to be used by BalanceTask """
        return self.sensors[4]

    def getPoleAngles(self):
        """ auxiliary access to just the pole angle(s), to be used by BalanceTask """
        return [self.sensors[0], self.sensors[2]]

