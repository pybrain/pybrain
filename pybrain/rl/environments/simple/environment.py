__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from scipy import random, zeros
from pybrain.rl.environments import Environment


class SimpleEnvironment(Environment):
    def __init__(self, dim=1):
        Environment.__init__(self)
        self.dim = dim
        self.indim = dim
        self.outdim = dim
        self.noise = None
        self.reset()

    def setNoise(self, variance):
        self.noise = variance

    def getSensors(self):
        if not self.updated:
            self.update()
        return self.state

    def performAction(self, action):
        self.updated = False
        self.action = action

    def update(self):
        self.state = [s + 0.1 * a for s, a in zip(self.state, self.action)]
        if self.noise:
            self.state += random.normal(0, self.noise, self.dim)

    def f(self, x):
        return [v ** 2 for v in x]

    def reset(self):
        self.state = random.uniform(2, 2, self.dim)

        self.action = zeros(self.dim, float)
        self.updated = True


