__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'


""" This example demonstrates how to use the discrete Reinforcement Learning
    algorithms (SARSA, Q, Q(lambda)) in a classical fully observable MDP 
    maze task. The goal point is the top right free field.
"""

from scipy import array
import pylab

from pybrain.rl.environments.mazes import Maze 
from pybrain.rl.learners.valuebased.interface import ActionValueTable
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Q, QLambda, SARSA
from pybrain.rl.explorers import BoltzmannExplorer
from pybrain.rl.experiments import Experiment
from pybrain.rl.environments import Task


class MazeTask(Task):
    """ We have to write our own MazeTask to make the problem fully observable MDP. """
    
    def getReward(self):
        """ compute and return the current reward (i.e. corresponding to the last action performed) """
        if self.env.goal == self.env.perseus:
            self.env.reset()
            reward = 1.
        else: 
            reward = 0.
        return reward
    
    def performAction(self, action):
        """ The action vector is stripped and the only element is cast to integer and given 
            to the super class. 
        """
        Task.performAction(self, int(action[0]))

    
    def getObservation(self):
        """ The agent receives its position in the maze, to make this a fully observable
            MDP problem.
        """
        obs = array([self.env.perseus[0] * self.env.mazeTable.shape[0] + self.env.perseus[1]])   
        return obs   



# create the maze with walls (1)
envmatrix = array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 0, 0, 1, 0, 0, 0, 0, 1],
                   [1, 0, 0, 1, 0, 0, 1, 0, 1],
                   [1, 0, 0, 1, 0, 0, 1, 0, 1],
                   [1, 0, 0, 1, 0, 1, 1, 0, 1],
                   [1, 0, 0, 0, 0, 0, 1, 0, 1],
                   [1, 1, 1, 1, 1, 1, 1, 0, 1],
                   [1, 0, 0, 0, 0, 0, 0, 0, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1]])

env = Maze(envmatrix, (7, 7))

# create task
task = MazeTask(env)

# create value table and initialize with ones
table = ActionValueTable(81, 4)
table.initialize(1.)

# create agent with controller and learner - use SARSA(), Q() or QLambda() here
learner = SARSA()

# standard exploration is e-greedy, but a different type can be chosen as well
# learner.explorer = BoltzmannExplorer()

agent = LearningAgent(table, learner)

# create experiment
experiment = Experiment(task, agent)

# prepare plotting
pylab.gray()
pylab.ion()

for i in range(1000):
    
    # interact with the environment (here in batch mode)
    experiment.doInteractions(100)
    agent.learn()
    agent.reset()
    
    # and draw the table
    pylab.pcolor(table.values.max(1).reshape(9,9))
    pylab.draw()

    

