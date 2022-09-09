from __future__ import print_function
#!/usr/bin/env python
"""
Illustrating how to use optimization algorithms in a reinforcement learning framework.
"""

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.utilities import fListToString
from pybrain.rl.environments.cartpole.balancetask import BalanceTask
from pybrain.tools.shortcuts import buildNetwork
from pybrain.optimization import HillClimber, CMAES #@UnusedImport
# from pybrain.rl.learners.continuous.policygradients import ENAC
# from pybrain.rl.agents.learning import LearningAgent
from pybrain.rl.agents import OptimizationAgent
from pybrain.rl.experiments import EpisodicExperiment


# any episodic task
task = BalanceTask()

# any neural network controller
net = buildNetwork(task.outdim, 1, task.indim)

# any optimization algorithm to be plugged in, for example:
# learner = CMAES(storeAllEvaluations = True)
# or:
learner = HillClimber(storeAllEvaluations = True)

# in a non-optimization case the agent would be a LearningAgent:
# agent = LearningAgent(net, ENAC())
# here it is an OptimizationAgent:
agent = OptimizationAgent(net, learner)

# the agent and task are linked in an Experiment
# and everything else happens under the hood.
exp = EpisodicExperiment(task, agent)
exp.doEpisodes(100)

print('Episodes learned from:', len(learner._allEvaluations))
n, fit = learner._bestFound()
print('Best fitness found:', fit)
print('with this network:')
print(n)
print('containing these parameters:')
print(fListToString(n.params, 4))
