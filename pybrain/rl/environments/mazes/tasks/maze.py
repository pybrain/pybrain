__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import array

from pomdp import POMDPTask
from pybrain.rl.environments.mazes import Maze
from pybrain.rl.environments.task import Task


class MazeTask(POMDPTask):
    """ a task corresponding to a maze environment """

    bangPenalty = 0
    defaultPenalty = 0
    finalReward = 1

    topology = None
    goal = None
    initPos = None
    mazeclass = Maze

    stochObs = 0
    stochAction = 0

    @property
    def noisy(self):
        return self.stochObs > 0

    def __init__(self, **args):
        self.setArgs(**args)
        Task.__init__(self, self.mazeclass(self.topology, self.goal, initPos=self.initPos,
                                           stochObs=self.stochObs, stochAction=self.stochAction))
        self.minReward = min(self.bangPenalty, self.defaultPenalty)
        self.reset()

    def getReward(self):
        if self.env.perseus == self.env.goal:
            return self.finalReward
        elif self.env.bang:
            return self.bangPenalty
        else:
            return self.defaultPenalty

    def isFinished(self):
        return self.env.perseus == self.env.goal or POMDPTask.isFinished(self)

    def __str__(self):
        return str(self.env)


class TrivialMaze(MazeTask):
    """
    #####
    #. *#
    #####
    """
    discount = 0.8
    initPos = [(1, 1)]
    topology = array([[1] * 5,
                      [1, 0, 0, 0, 1],
                      [1] * 5, ])
    goal = (1, 3)

