__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import array

from maze import MazeTask
from pybrain.rl.environments.mazes import PolarMaze


class EightyNineStateMaze(MazeTask):
    """
    #########
    ##     ##
    #  # #  #
    ## # # ##
    #  # # *#
    ##     ##
    #########
    """

    mazeclass = PolarMaze
    topology = array([[1]*9,
                      [1, 1, 0, 0, 0, 0, 0, 1, 1],
                      [1, 0, 0, 1, 0, 1, 0, 0, 1],
                      [1, 1, 0, 1, 0, 1, 0, 1, 1],
                      [1, 0, 0, 1, 0, 1, 0, 0, 1],
                      [1, 1, 0, 0, 0, 0, 0, 1, 1],
                      [1]*9])
    goal = (2, 7)
    stochAction = 0.1
    stochObs = 0.1
