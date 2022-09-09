__author__ = 'Tom Schaul, tom@idsia.ch'

from random import random, choice
from scipy import zeros

from pybrain.utilities import Named
from pybrain.rl.environments.environment import Environment

# TODO: mazes can have any number of dimensions?


class Maze(Environment, Named):
    """ 2D mazes, with actions being the direction of movement (N,E,S,W)
    and observations being the presence of walls in those directions.

    It has a finite number of states, a subset of which are potential starting states (default: all except goal states).
    A maze can have absorbing states, which, when reached end the episode (default: there is one, the goal).

    There is a single agent walking around in the maze (Theseus).
    The movement can succeed or not, or be stochastically determined.
    Running against a wall does not get you anywhere.

    Every state can have an an associated reward (default: 1 on goal, 0 elsewhere).
    The observations can be noisy.
    """

    # table of booleans
    mazeTable = None

    # single goal
    goal = None

    # current state
    perseus = None

    # list of possible initial states
    initPos = None

    # directions
    N = (1, 0)
    S = (-1, 0)
    E = (0, 1)
    W = (0, -1)

    allActions = [N, E, S, W]

    # stochasticity
    stochAction = 0.
    stochObs = 0.

    def __init__(self, topology, goal, **args):
        self.setArgs(**args)
        self.mazeTable = topology
        self.goal = goal
        if self.initPos == None:
            self.initPos = self._freePos()
            self.initPos.remove(self.goal)
        self.reset()

    def reset(self):
        """ return to initial position (stochastically): """
        self.bang = False
        self.perseus = choice(self.initPos)

    def _freePos(self):
        """ produce a list of the free positions. """
        res = []
        for i, row in enumerate(self.mazeTable):
            for j, p in enumerate(row):
                if p == False:
                    res.append((i, j))
        return res

    def _moveInDir(self, pos, dir):
        """ the new state after the movement in one direction. """
        return (pos[0] + dir[0], pos[1] + dir[1])

    def performAction(self, action):
        if self.stochAction > 0:
            if random() < self.stochAction:
                action = choice(list(range(len(self.allActions))))
        tmp = self._moveInDir(self.perseus, self.allActions[action])
        if self.mazeTable[tmp] == False:
            self.perseus = tmp
            self.bang = False
        else:
            self.bang = True

    def getSensors(self):
        obs = zeros(4)
        for i, a in enumerate(Maze.allActions):
            obs[i] = self.mazeTable[self._moveInDir(self.perseus, a)]
        if self.stochObs > 0:
            for i in range(len(obs)):
                if random() < self.stochObs:
                    obs[i] = not obs[i]
        return obs

    def __str__(self):
        """ Ascii representation of the maze, with the current state """
        s = ''
        for r, row in reversed(list(enumerate(self.mazeTable))):
            for c, p in enumerate(row):
                if (r, c) == self.goal:
                    s += '*'
                elif (r, c) == self.perseus:
                    s += '@'
                elif p == True:
                    s += '#'
                else:
                    s += ' '
            s += '\n'
        return s


