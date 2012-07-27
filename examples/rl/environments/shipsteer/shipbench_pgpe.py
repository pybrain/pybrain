#!/usr/bin/env python
#########################################################################
# Reinforcement Learning with PGPE on the ShipSteering Environment
#
# Requirements: pylab (for plotting only). If not available, comment the
# last 3 lines out
# Author: Frank Sehnke, sehnke@in.tum.de
#########################################################################
__author__ = "Martin Felder, Frank Sehnke"
__version__ = '$Id$' 

from pybrain.tools.example_tools import ExTools
from pybrain.structure.modules.tanhlayer import TanhLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.rl.environments.shipsteer import ShipSteeringEnvironment
from pybrain.rl.environments.shipsteer import GoNorthwardTask
from pybrain.rl.agents import OptimizationAgent
from pybrain.optimization import PGPE 
from pybrain.rl.experiments import EpisodicExperiment

batch=1 #number of samples per learning step
prnts=50 #number of learning steps after results are printed
epis=2000/batch/prnts #number of roleouts
numbExp=10 #number of experiments
et = ExTools(batch, prnts) #tool for printing and plotting

env = None
for runs in range(numbExp):
    # create environment
    #Options: Bool(OpenGL), Bool(Realtime simu. while client is connected), ServerIP(default:localhost), Port(default:21560)
    if env != None: env.closeSocket()
    env = ShipSteeringEnvironment()
    # create task
    task = GoNorthwardTask(env,maxsteps = 500)
    # create controller network
    net = buildNetwork(task.outdim, task.indim, outclass=TanhLayer)
    # create agent with controller and learner (and its options)
    agent = OptimizationAgent(net, PGPE(learningRate = 0.3,
                                    sigmaLearningRate = 0.15,
                                    momentum = 0.0,
                                    epsilon = 2.0,
                                    rprop = False,
                                    storeAllEvaluations = True))
    et.agent = agent
    #create experiment
    experiment = EpisodicExperiment(task, agent)

    #Do the experiment
    for updates in range(epis):
        for i in range(prnts):
            experiment.doEpisodes(batch)
        et.printResults((agent.learner._allEvaluations)[-50:-1], runs, updates)
    et.addExps()
et.showExps()
#To view what the simulation is doing at the moment set the environment with True, go to pybrain/rl/environments/ode/ and start viewer.py (python-openGL musst be installed, see PyBrain documentation)
