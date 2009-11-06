#########################################################################
# Reinforcement Learning with Policy Gradients on the SimpleEnvironment 
#
# SimpleEnvironment is a one-dimensional quadratic function with its
# maximum at x=0. Additionally, noise can be added (setNoise(variance)).
# The Agent can make steps in either direction and has receives reward 
# equal to the negative function value. The optimal parameter is approx.
# -2.6
# Requirements: pylab (for plotting only). If not available, comment the
# 5 lines marked as "for plotting"
# Author: Frank Sehnke, sehnke@in.tum.de
#########################################################################

from pybrain.tools.shortcuts import buildNetwork
from pybrain.rl.environments.flexcube import FlexCubeEnvironment, GrowTask
from pybrain.rl.agents import LearningAgent
from pybrain.optimization import PGPE 
from pybrain.rl.experiments import EpisodicExperiment
from scipy import mean

# for plotting
from pylab import ion,clf, plot, draw 

ion() 

# create environment
env = FlexCubeEnvironment()
# create task
task = GrowTask(env)
# create controller network (flat network)
net = buildNetwork(32, 10, 12)
# create agent with controller and learner
agent = LearningAgent(net, PGPE(learningRate=0.05,
                                sigmaLearningRate=0.1,
                                momentum=0.0,
                                epsilon=2.0,
                                #rprop = True,
                                ))
# TODO: this method does not work with the new framework anymore, is it
# correct just to leave it out?
# agent.learner.initSigmas()

sr = []

experiment = EpisodicExperiment(task, agent)
for updates in range(1000):
    # training step
    for i in range(5):
        experiment.doEpisodes(10)
        agent.learn()
        print "parameters:", agent.module.params
        agent.reset()
        
    # learning step
    agent.disableLearning()
    experiment.doEpisodes(50)
    # append mean reward to sr array
    ret = []
    for n in range(agent.history.getNumSequences()):
        state, action, reward, _ = agent.history.getSequence(n)
        ret.append( sum(reward, 0).item() )
    sr.append(mean(ret))
        
    agent.enableLearning()
    agent.reset()

    # for plotting
    clf()
    plot(sr)
    draw()            
