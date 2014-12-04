from __future__ import print_function

#!/usr/bin/env python
#########################################################################
# Reinforcement Learning with SPE on the ShipSteering Environment
#
# Requirements:
#   pybrain (tested on rev. 1195, ship env rev. 1202)
# Synopsis:
#   shipbenchm.py [<True|False> [logfile]]
#       (first argument is graphics flag)
#########################################################################
__author__ = "Martin Felder, Thomas Rueckstiess"
__version__ = '$Id$'

#---
# default backend GtkAgg does not plot properly on Ubuntu 8.04
import matplotlib
matplotlib.use('TkAgg')
#---

from pybrain.rl.environments.shipsteer import ShipSteeringEnvironment
from pybrain.rl.environments.shipsteer import GoNorthwardTask
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners.directsearch.enac import ENAC
from pybrain.rl.experiments.episodic import EpisodicExperiment

from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.plotting import MultilinePlotter

from pylab import figure, ion
from scipy import mean
import sys

if len(sys.argv) > 1:
    useGraphics = eval(sys.argv[1])
else:
    useGraphics = False

# create task
env=ShipSteeringEnvironment()
maxsteps = 500
task = GoNorthwardTask(env=env, maxsteps = maxsteps)
# task.env.setRenderer( CartPoleRenderer())

# create controller network
#net = buildNetwork(task.outdim, 7, task.indim, bias=True, outputbias=False)
net = buildNetwork(task.outdim, task.indim, bias=False)
#net.initParams(0.0)


# create agent
learner = ENAC()
learner.gd.rprop = True
# only relevant for RP
learner.gd.deltamin = 0.0001
#agent.learner.gd.deltanull = 0.05
# only relevant for BP
learner.gd.alpha = 0.01
learner.gd.momentum = 0.9

agent = LearningAgent(net, learner)
agent.actaspg = False

# create experiment
experiment = EpisodicExperiment(task, agent)

# print weights at beginning
print(agent.module.params)

rewards = []
if useGraphics:
    figure()
    ion()
    pl = MultilinePlotter(autoscale=1.2, xlim=[0, 50], ylim=[0, 1])
    pl.setLineStyle(linewidth=2)

# queued version
# experiment._fillQueue(30)
# while True:
#     experiment._stepQueueLoop()
#     # rewards.append(mean(agent.history.getSumOverSequences('reward')))
#     print agent.module.getParameters(),
#     print mean(agent.history.getSumOverSequences('reward'))
#     clf()
#     plot(rewards)

# episodic version
x = 0
batch = 30 #number of samples per gradient estimate (was: 20; more here due to stochastic setting)
while x<5000:
#while True:
    experiment.doEpisodes(batch)
    x += batch
    reward = mean(agent.history.getSumOverSequences('reward'))*task.rewardscale
    if useGraphics:
        pl.addData(0,x,reward)
    print(agent.module.params)
    print(reward)
    #if reward > 3:
    #    pass
    agent.learn()
    agent.reset()
    if useGraphics:
        pl.update()


if len(sys.argv) > 2:
    agent.history.saveToFile(sys.argv[1], protocol=-1, arraysonly=True)
if useGraphics:
    pl.show( popup = True)

#To view what the simulation is doing at the moment set the environment with True, go to pybrain/rl/environments/ode/ and start viewer.py (python-openGL musst be installed, see PyBrain documentation)

## performance:
## experiment.doEpisodes(5) * 100 without weave:
##    real    2m39.683s
##    user    2m33.358s
##    sys     0m5.960s
## experiment.doEpisodes(5) * 100 with weave:
##real    2m41.275s
##user    2m35.310s
##sys     0m5.192s
##
