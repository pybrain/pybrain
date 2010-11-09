#!/usr/bin/env python
#########################################################################
# Reinforcement Learning with REINFORCE on the CartPoleEnvironment 
#
# Requirements: pylab (for plotting only). If not available, comment the
# last 3 lines out
#########################################################################
__author__ = "Thomas Rueckstiess, Frank Sehnke"
__version__ = '$Id$' 

from pybrain.tools.example_tools import ExTools
from pybrain.tools.shortcuts import buildNetwork
from pybrain.rl.environments.cartpole import CartPoleEnvironment, BalanceTask
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Reinforce
from pybrain.rl.experiments import EpisodicExperiment

batch=50 #number of samples per learning step
prnts=4 #number of learning steps after results are printed
epis=4000/batch/prnts #number of roleouts
numbExp=10 #number of experiments
et = ExTools(batch, prnts, kind = "learner") #tool for printing and plotting

for runs in range(numbExp):
    # create environment
    env = CartPoleEnvironment()    
    # create task
    task = BalanceTask(env, 200, desiredValue=None)
    # create controller network
    net = buildNetwork(4, 1, bias=False)
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
