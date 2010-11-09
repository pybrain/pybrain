#!/usr/bin/env python
#########################################################################
# Reinforcement Learning with PGPE on the FlexCube Environment 
#
# The FlexCube Environment is a Mass-Spring-System composed of 8 mass points.
# These resemble a cube with flexible edges.
#
# Control/Actions:
# The agent can control the 12 equilibrium edge lengths. 
#
# A wide variety of sensors are available for observation and reward:
# - 12 edge lengths
# - 12 wanted edge lengths (the last action)
# - vertexes contact with floor
# - vertexes min height (distance of closest vertex to the floor)
# - distance to origin
# - distance and angle to target
#
# Task available are:
# - GrowTask, agent has to maximize the volume of the cube
# - JumpTask, agent has to maximize the distance of the lowest mass point during the episode
# - WalkTask, agent has to maximize the distance to the starting point
# - WalkDirectionTask, agent has to minimize the distance to a target point.
# - TargetTask, like the previous task but with several target points
# 
# Requirements: pylab (for plotting only). If not available, comment the
# last 3 lines out
# Author: Frank Sehnke, sehnke@in.tum.de
#########################################################################
__author__ = "Frank Sehnke"
__version__ = '$Id$' 

from pybrain.tools.example_tools import ExTools
from pybrain.structure.modules.tanhlayer import TanhLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.rl.environments.flexcube import FlexCubeEnvironment, WalkTask
from pybrain.rl.agents import OptimizationAgent
from pybrain.optimization import PGPE 
from pybrain.rl.experiments import EpisodicExperiment

hiddenUnits = 4
batch=1 #number of samples per learning step
prnts=1 #number of learning steps after results are printed
epis=5000000/batch/prnts #number of roleouts
numbExp=10 #number of experiments
et = ExTools(batch, prnts) #tool for printing and plotting

env = None
for runs in range(numbExp):
    # create environment
    #Options: Bool(OpenGL), Bool(Realtime simu. while client is connected), ServerIP(default:localhost), Port(default:21560)
    if env != None: env.closeSocket()
    env = FlexCubeEnvironment()
    # create task
    task = WalkTask(env)
    # create controller network
    net = buildNetwork(len(task.getObservation()), hiddenUnits, env.actLen, outclass=TanhLayer)    
    # create agent with controller and learner (and its options)
    agent = OptimizationAgent(net, PGPE(storeAllEvaluations = True))
    et.agent = agent
    # create the experiment
    experiment = EpisodicExperiment(task, agent)

    #Do the experiment
    for updates in range(epis):
        for i in range(prnts):
            experiment.doEpisodes(batch)
        et.printResults((agent.learner._allEvaluations)[-50:-1], runs, updates)
    et.addExps()
et.showExps()
#To view what the simulation is doing at the moment, go to pybrain/rl/environments/flexcube/ and start renderer.py (python-openGL musst be installed)
