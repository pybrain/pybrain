#!/usr/bin/env python
#########################################################################
# Reinforcement Learning with PGPE on the CCRL ODE Environment 
#
# The CCRL robot is a body structure with 2x 7 DoF Arms.
# Complex grasping tasks can be learned with this environment.
#
# Control/Actions:
# The agent can control all 14 DOF of the robot arms plus the 2 hands. 
#
# A wide variety of sensors are available for observation and reward:
# - 16 angles of joints
# - 16 angle velocitys of joints
# - Number of hand parts that have contact to target object
# - collision with table
# - distance of hand to target
# - angle of hand to horizontal and vertical plane
#
# Task available are:
# - Grasp Task, agent has to get hold of the object with avoiding collision with table
# 
# Requirements: pylab (for plotting only). If not available, comment the
# last 3 lines out
# Author: Frank Sehnke, sehnke@in.tum.de
#########################################################################
__author__ = "Frank Sehnke"
__version__ = '$Id$' 

from pybrain.tools.example_tools import ExTools
from pybrain.rl.environments.ode import CCRLEnvironment
from pybrain.rl.environments.ode.tasks import CCRLGlasTask
from pybrain.structure.modules.tanhlayer import TanhLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.rl.agents import OptimizationAgent
from pybrain.optimization import PGPE
from pybrain.rl.experiments import EpisodicExperiment

hiddenUnits = 4
batch=1 #number of samples per learning step
prnts=1 #number of learning steps after results are printed
epis=2000/batch/prnts #number of roleouts
numbExp=10 #number of experiments
et = ExTools(batch, prnts) #tool for printing and plotting

env = None
for runs in range(numbExp):
    # create environment
    #Options: XML-Model, Bool(OpenGL), Bool(Realtime simu. while client is connected), ServerIP(default:localhost), Port(default:21560)
    if env != None: env.closeSocket()
    env = CCRLEnvironment()
    # create task
    task = CCRLGlasTask(env)
    # create controller network
    net = buildNetwork(len(task.getObservation()), hiddenUnits, env.actLen, outclass=TanhLayer) #, hiddenUnits    
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
#To view what the simulation is doing at the moment, go to pybrain/rl/environments/ode/ and start viewer.py (python-openGL musst be installed, see PyBrain documentation)
