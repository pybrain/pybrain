__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from matplotlib.mlab import rk4
from math import sin, cos
import time
from scipy import eye, matrix, random, asarray

from pybrain.rl.environments.graphical import GraphicalEnvironment


class CartPoleEnvironment(GraphicalEnvironment):
    """ This environment implements the cart pole balancing benchmark, as stated in:
        Riedmiller, Peters, Schaal: "Evaluation of Policy Gradient Methods and
        Variants on the Cart-Pole Benchmark". ADPRL 2007.
        It implements a set of differential equations, solved with a 4th order
        Runge-Kutta method.
    """

    indim = 1
    outdim = 4

    # some physical constants
    g = 9.81
    l = 0.5
    mp = 0.1
    mc = 1.0
    dt = 0.02

    randomInitialization = True

    def __init__(self, polelength=None):
        GraphicalEnvironment.__init__(self)
        if polelength != None:
            self.l = polelength

        # initialize the environment (randomly)
        self.reset()
        self.action = 0.0
        self.delay = False

    def getSensors(self):
        """ returns the state one step (dt) ahead in the future. stores the state in
            self.sensors because it is needed for the next calculation. The sensor return
            vector has 4 elements: theta, theta', s, s' (s being the distance from the
            origin).
        """
        return asarray(self.sensors)

    def performAction(self, action):
        """ stores the desired action for the next runge-kutta step.
        """
        self.action = action
        self.step()

    def step(self):
        self.sensors = rk4(self._derivs, self.sensors, [0, self.dt])
        self.sensors = self.sensors[-1]
        if self.hasRenderer():
            self.getRenderer().updateData(self.sensors)
            if self.delay:
                time.sleep(0.05)

    def reset(self):
        """ re-initializes the environment, setting the cart back in a random position.
        """
        if self.randomInitialization:
            angle = random.uniform(-0.2, 0.2)
            pos = random.uniform(-0.5, 0.5)
        else:
            angle = -0.2
            pos = 0.2
        self.sensors = (angle, 0.0, pos, 0.0)

    def _derivs(self, x, t):
        """ This function is needed for the Runge-Kutta integration approximation method. It calculates the
            derivatives of the state variables given in x. for each variable in x, it returns the first order
            derivative at time t.
        """
        F = self.action
        (theta, theta_, _s, s_) = x
        u = theta_
        sin_theta = sin(theta)
        cos_theta = cos(theta)
        mp = self.mp
        mc = self.mc
        l = self.l
        u_ = (self.g * sin_theta * (mc + mp) - (F + mp * l * theta ** 2 * sin_theta) * cos_theta) / (4 / 3 * l * (mc + mp) - mp * l * cos_theta ** 2)
        v = s_
        v_ = (F - mp * l * (u_ * cos_theta - (s_ ** 2 * sin_theta))) / (mc + mp)
        return (u, u_, v, v_)

    def getPoleAngles(self):
        """ auxiliary access to just the pole angle(s), to be used by BalanceTask """
        return [self.sensors[0]]

    def getCartPosition(self):
        """ auxiliary access to just the cart position, to be used by BalanceTask """
        return self.sensors[2]



class CartPoleLinEnvironment(CartPoleEnvironment):
    """ This is a linearized implementation of the cart-pole system, as described in
    Peters J, Vijayakumar S, Schaal S (2003) Reinforcement learning for humanoid robotics.
    Polelength is fixed, the order of sensors has been changed to the above."""

    tau = 1. / 60.   # sec

    def __init__(self, **kwargs):
        CartPoleEnvironment.__init__(self, **kwargs)
        nu = 13.2 #  sec^-2
        tau = self.tau

        # linearized movement equations
        self.A = matrix(eye(4))
        self.A[0, 1] = tau
        self.A[2, 3] = tau
        self.A[1, 0] = nu * tau
        self.b = matrix([0.0, nu * tau / 9.80665, 0.0, tau])


    def step(self):
        self.sensors = random.normal(loc=self.sensors * self.A + self.action * self.b, scale=0.001).flatten()
        if self.hasRenderer():
            self.getRenderer().updateData(self.sensors)
            if self.delay:
                time.sleep(self.tau)

    def reset(self):
        """ re-initializes the environment, setting the cart back in a random position.
        """
        self.sensors = random.normal(scale=0.1, size=4)

    def getSensors(self):
        return self.sensors.flatten()

    def getPoleAngles(self):
        """ auxiliary access to just the pole angle(s), to be used by BalanceTask """
        return [self.sensors[0]]

    def getCartPosition(self):
        """ auxiliary access to just the cart position, to be used by BalanceTask """
        return self.sensors[2]

