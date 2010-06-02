#!/usr/bin/env python
#########################################################################
# Reinforcement Learning with PGPE on the Johnnie Environment 
#
# The Johnnie robot is a body structure with 11 DoF .
# Complex balancing tasks can be learned with this environment.
#
# Control/Actions:
# The agent can control all 11 DOF of the robot. 
#
# A wide variety of sensors are available for observation and reward:
# - 11 angles of joints
# - 11 angle velocitys of joints
# - Number of foot parts that have contact to floor
# - Height sensor in head for reward calculation
# - Rotation sensor in 3 dimesnions
#
# Task available are:
# - StandTask, agent has not to fall by himself
# - Robust standing Task, agent has not to fall even then hit by reasonable random forces
# - JumpTask, agent has to maximize the head-vertical position during the episode
# 
# Requirements: pylab (for plotting only). If not available, comment the
# last 3 lines out
# Author: Frank Sehnke, sehnke@in.tum.de
#########################################################################
__author__ = "Frank Sehnke"
__version__ = '$Id$' 

from pybrain.tools.example_tools import ExTools
from pybrain.rl.environments.ode import JohnnieEnvironment
from pybrain.rl.environments.ode.tasks import StandingTask
from pybrain.structure.modules.tanhlayer import TanhLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Reinforce
from pybrain.rl.experiments import EpisodicExperiment

hiddenUnits = 4
batch=2 #number of samples per learning step
prnts=1 #number of learning steps after results are printed
epis=5000000/batch/prnts #number of roleouts
numbExp=10 #number of experiments
et = ExTools(batch, prnts, kind = "learner")#tool for printing and plotting

for runs in range(numbExp):
    # create environment
    #Options: Bool(OpenGL), Bool(Realtime simu. while client is connected), ServerIP(default:localhost), Port(default:21560)
    env = JohnnieEnvironment() 
    # create task
    task = StandingTask(env)
    # create controller network
    net = buildNetwork(len(task.getObservation()), hiddenUnits, env.actLen, outclass=TanhLayer)    
    # create agent with controller and learner (and its options)
    agent = LearningAgent(net, Reinforce())
    et.agent = agent
    # create the experiment
    experiment = EpisodicExperiment(task, agent)

    #Do the experiment
    for updates in range(epis):
        for i in range(prnts):
            experiment.doEpisodes(batch)
        state, action, reward = agent.learner.dataset.getSequence(agent.learner.dataset.getNumSequences()-1)
        et.printResults(reward.sum(), runs, updates)
    et.addExps()
et.showExps()
#To view what the simulation is doing at the moment, go to pybrain/rl/environments/ode/ and start viewer.py (python-openGL musst be installed, see PyBrain documentation)
