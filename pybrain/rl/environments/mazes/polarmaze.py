__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import zeros
from random import choice, random

from maze import Maze


class PolarMaze(Maze):
    """ Mazes with the emphasis on Perseus: allow him to turn, go forward or backward.
    Thus there are 4 states per position.
    """

    actions = 5

    Stay = 0
    Forward = 1
    TurnAround = 2
    TurnLeft = 3
    TurnRight = 4

    allActions = [Stay, Forward, TurnAround, TurnLeft, TurnRight]

    def reset(self):
        Maze.reset(self)
        self.perseusDir = choice(range(4))

    def performAction(self, action):
        if self.stochAction > 0:
            if random() < self.stochAction:
                action = choice(range(len(PolarMaze.allActions)))
        act = PolarMaze.allActions[action]
        self.bang = False
        if act == self.Forward:
            tmp = self._moveInDir(self.perseus, Maze.allActions[self.perseusDir])
            if self.mazeTable[tmp] == False:
                self.perseus = tmp
            else:
                self.bang = True
        elif act == self.TurnLeft:
            self.perseusDir = (self.perseusDir + 1) % 4
        elif act == self.TurnRight:
            self.perseusDir = (self.perseusDir - 1) % 4
        elif act == self.TurnAround:
            self.perseusDir = (self.perseusDir + 2) % 4

    def getSensors(self):
        obs = Maze.getSensors(self)
        res = zeros(4)
        res[:4 - self.perseusDir] = obs[self.perseusDir:]
        res[4 - self.perseusDir:] = obs[:self.perseusDir]
        return res

    def __str__(self):
        return Maze.__str__(self) + '(dir:' + str(self.perseusDir) + ')'
