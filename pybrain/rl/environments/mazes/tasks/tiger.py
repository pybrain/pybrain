__author__ = 'Tom Schaul, tom@idsia.ch'

from random import choice, random
from scipy import array

from pomdp import POMDPTask


class TigerTask(POMDPTask):
    """ two doors, behind one is a tiger - we can listen or open. """

    observations = 2
    actions = 3
    minReward = -100
    discount = 0.75

    # allowed actions:
    Listen = 0
    OpenLeft = 1
    OpenRight = 2

    # stochsticity on the observation
    stochObs = 0.15

    def reset(self):
        self.steps = 0
        self.cumreward = 0
        self.tigerLeft = choice([True, False])
        self.done = False
        self.nextReward = -1

    def performAction(self, action):
        self.steps += 1
        if action != self.Listen:
            self.done = True
            if ((action == self.OpenLeft and self.tigerLeft)
                or (action == self.OpenRight and not self.tigerLeft)):
                self.nextReward = -100
            else:
                self.nextReward = 10

    def getObservation(self):
        """ do we think we heard something on the left or on the right?  """
        if self.tigerLeft:
            obs = array([1, 0])
        else:
            obs = array([0, 1])
        if random() < self.stochObs:
            obs = 1 - obs
        return obs

    def isFinished(self):
        return self.done or POMDPTask.isFinished(self)

    def getReward(self):
        return self.nextReward
