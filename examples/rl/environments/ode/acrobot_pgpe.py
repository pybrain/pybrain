#!/usr/bin/env python
#########################################################################
# Reinforcement Learning with PGPE on the Acrobot Environment 
#
# The Acrobot Environment is a 1 DoF system.
# The goal is to swing up the pole and balance it.
# The motor is underpowered so that the pole can not go directly to the upright position.
# It has to swing several times to gain enough momentum.
#
# Control/Actions:
# The agent can control 1 joint. 
#
# Requirements: pylab (for plotting only). If not available, comment the
# last 3 lines out
# Author: Frank Sehnke, sehnke@in.tum.de
#########################################################################
__author__ = "Frank Sehnke"
__version__ = '$Id$' 

from pybrain.tools.example_tools import ExTools
from pybrain.rl.environments.ode import AcrobotEnvironment
from pybrain.rl.environments.ode.tasks import GradualRewardTask
from pybrain.structure.modules.tanhlayer import TanhLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.rl.agents import OptimizationAgent
from pybrain.optimization import PGPE
from pybrain.rl.experiments import EpisodicExperiment
from time import sleep

batch=1 #number of samples per learning step
prnts=1 #number of learning steps after results are printed
epis=4000/batch/prnts #number of roleouts
numbExp=40 #number of experiments
et = ExTools(batch, prnts) #tool for printing and plotting

env = None
for runs in range(numbExp):
    # create environment
    #Options: Bool(OpenGL), Bool(Realtime simu. while client is connected), ServerIP(default:localhost), Port(default:21560)
    if env != None: env.closeSocket()
    env = AcrobotEnvironment()
    # create task
    task = GradualRewardTask(env)
    # create controller network
    net = buildNetwork(len(task.getObservation()), env.actLen, outclass=TanhLayer)    
    # create agent with controller and learner (and its options)
    agent = OptimizationAgent(net, PGPE(storeAllEvaluations = True,
                                    learningRate = 0.05,
                                    sigmaLearningRate = 0.025,
                                    momentum = 0.0,
                                    epsilon = 6.0,
                                    rprop = False,))
    et.agent = agent
    #agent.learner.bestEvaluation = 1500
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
