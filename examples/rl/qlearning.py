from pybrain.rl.environments.mazes import Maze 
from pybrain.structure.modules import ActionValueTable
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Q, QLambda, SARSA
from pybrain.rl.explorers import EpsilonGreedyExplorer, BoltzmannExplorer
from pybrain.rl.experiments import ContinuousExperiment
from pybrain.rl import Task

from scipy import *

import sys, time
import pylab

class MazeTask(Task):
    def getReward(self):
        """ compute and return the current reward (i.e. corresponding to the last action performed) """
        if self.env.goal == self.env.perseus:
            self.env.reset()
            reward = 1.
        else: 
            reward = 0.
        return reward
    
    def performAction(self, action):
        """ a filtered mapping towards performAction of the underlying environment. """
        Task.performAction(self, int(action[0]))

    
    def getObservation(self):
        """ a filtered mapping to getSample of the underlying environment. """
        obs = array([self.env.perseus[0] * self.env.mazeTable.shape[0] + self.env.perseus[1]])   
        return obs   

# create environment
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

# create the ActionValueTable
table = ActionValueTable(81, 4)
table.initialize(1)

# create agent with controller and learner
agent = LearningAgent(table, QLambda())

experiment = ContinuousExperiment(task, agent)

pylab.gray()
pylab.ion()

for i in range(100000):
    experiment.doInteractionsAndLearn()
    
    if i % 100 == 0:
        pylab.pcolor(table.values.max(1).reshape(9,9))
        pylab.draw()
        agent.reset()
    

