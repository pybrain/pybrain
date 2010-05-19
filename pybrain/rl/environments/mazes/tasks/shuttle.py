__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import array, zeros
from random import random

from maze import MazeTask
from pybrain.rl.environments.mazes import PolarMaze


class ShuttleDocking(MazeTask):
    """
    #######
    #.   *#
    #######

    The spaceship needs to dock backwards into the goal station.
    """

    actions = 3
    observations = 5
    discount = 0.95

    mazeclass = PolarMaze

    finalReward = 10
    bangPenalty = -3

    initPos = [(1, 1)]
    topology = array([[1] * 7,
                      [1, 0, 0, 0, 0, 0, 1],
                      [1] * 7, ])
    goal = (1, 5)

    Backup = 0
    Forward = 1
    TurnAround = 2

    def reset(self):
        MazeTask.reset(self)
        self.env.perseusDir = 1

    def getObservation(self):
        """ inold, seeold, black, seenew, innew """
        res = zeros(5)
        if self.env.perseus == self.env.goal:
            res[4] = 1
        elif self.env.perseus == self.env.initPos[0]:
            res[0] = 1
        elif self.env.perseus[1] == 3:
            if random() > 0.7:
                res[self.env.perseusDir] = 1
            else:
                res[(self.env.perseusDir + 2) % 4] = 1
        else:
            res[(self.env.perseusDir + 2) % 4] = 1
        return res

    def performAction(self, action):
        self.steps += 1
        if action == self.TurnAround:
            self._turn()
        elif action == self.Forward:
            self._forward()
        else: # noisy backup
            r = random()
            if self.env.perseus[1] == 3:
                # in space
                if r < 0.1:
                    self._turn()
                elif r < 0.9:
                    self._backup()
            elif ((self.env.perseus[1] == 2 and self.env.perseusDir == 3) or
                  (self.env.perseus[1] == 4 and self.env.perseusDir == 1)):
                # close to station, front to station
                if r < 0.3:
                    self._turn()
                elif r < 0.6:
                    self._backup()
            else:
                # close to station, back to station
                if r < 0.7:
                    self._backup()

    def _backup(self):
        self.env.performAction(PolarMaze.TurnAround)
        self.env.performAction(PolarMaze.Forward)
        self.env.performAction(PolarMaze.TurnAround)

    def _turn(self):
        self.env.performAction(PolarMaze.TurnAround)

    def _forward(self):
        old = self.env.perseus
        self.env.performAction(PolarMaze.TurnAround)
        if self.env.perseus == self.env.goal or self.env.perseus == self.env.initPos[0]:
            self.env.perseus = old
            self.env.bang = True
