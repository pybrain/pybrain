#########################################################################
# Reinforcement Learning with Policy Gradients on the SimpleEnvironment 
#
# SimpleEnvironment is a one-dimensional quadratic function with its
# maximum at x=0. Additionally, noise can be added (setNoise(variance)).
# The Agent can make steps in either direction and has receives reward 
# equal to the negative function value. The optimal parameter is -10.0

# Requirements: pylab (for plotting only). If not available, comment the
# 9 lines total marked as "for plotting"
# Author: Frank Sehnke, sehnke@in.tum.de
#########################################################################

from pybrain.tools.shortcuts import buildNetwork
from pybrain.rl.environments.cartpole import CartPoleEnvironment, BalanceTask
from pybrain.rl.agents import LearningAgent
from pybrain.optimization import PGPE
from pybrain.rl.experiments import EpisodicExperiment
from scipy import random

numbExp=12
for runs in range(numbExp):
    env = CartPoleEnvironment()    
    # create task
    task = BalanceTask(env, 200)
    # create controller network
    net = buildNetwork(4, 1, bias=False)
    # create agent with controller and learner
    agent = LearningAgent(net, PGPE())
    # learning options
    agent.learner.gd.alpha = 0.05
    agent.learner.gdSig.alpha = 0.1
    agent.learner.gd.momentum = 0.9
    agent.learner.epsilon = 6.0
    agent.learner.initSigmas()
    # agent.learner.rprop = True
    experiment = EpisodicExperiment(task, agent)
    batch=16
    prnts=10
    epis=50000/batch/prnts
    save=False

    rl=[]
    for updates in range(epis):
        for i in range(prnts):
            experiment.doEpisodes(batch)
            agent.learn()
            agent.reset()
        print "Parameters: ", agent.learner.original
        print "Epsilon   : ", agent.learner.sigList
        print "Step: ", runs, "/", (updates+1)*batch*prnts, "Best: ", agent.learner.best, "Base: ", agent.learner.baseline, "Reward: ", agent.learner.reward   
        print ""
        rl.append(float(agent.learner.baseline))
    if save:
        fnStart="dataCartMom"
        fnExp=repr(int(agent.learner.gd.alpha*100))+"m"+repr(int(agent.learner.gdSig.alpha*100))+"s"+repr(batch/2)+"b"+repr(int(agent.learner.epsilon*10))+"e"
        fnIdent="PGPE"+repr(int(random.random()*1000000.0))
        filename=fnStart+fnExp+fnIdent+".dat"
        file = open(filename,"w")
        rlLen=len(rl)
        for i in range(rlLen):
            file.write(repr((i+1)*batch*prnts)+"\n")
            file.write(repr(rl[i])+"\n")
        file.close()       
