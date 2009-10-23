#########################################################################
# Reinforcement Learning with PGPE on the CartPoleEnvironment 
#
# Requirements: pylab (for plotting only). If not available, comment the
# 9 lines total marked as "for plotting"
# Author: Frank Sehnke, sehnke@in.tum.de
#########################################################################

from scipy import random

from pybrain.tools.shortcuts import buildNetwork
from pybrain.rl.environments.cartpole import CartPoleEnvironment, BalanceTask
from pybrain.rl.agents import OptimizationAgent
from pybrain.optimization import PGPE
from pybrain.rl.experiments import EpisodicExperiment

numbExp=12
for runs in range(numbExp):
    env = CartPoleEnvironment()    
    # create task
    task = BalanceTask(env, 200)
    # create controller network
    net = buildNetwork(4, 1, bias=False)
    # create agent with controller and learner (and its options)
    agent = OptimizationAgent(net, PGPE(learningRate = 0.5,
                                        sigmaLearningRate = 0.1,
                                        momentum = 0.0,
                                        epsilon = 2.0,
                                        #rprop = True,
                                        ))
    
    experiment = EpisodicExperiment(task, agent)
    batch=2
    prnts=80
    epis=2000/batch/prnts
    save=False

    rl=[]
    for updates in range(epis):
        for i in range(prnts):
            experiment.doEpisodes(batch)
        print "Parameters: ", agent.learner.current
        print "Epsilon   : ", agent.learner.sigList
        print "Step: ", runs, "/", (updates+1)*batch*prnts, "Best: ", agent.learner.bestEvaluation, 
        print "Base: ", agent.learner.baseline   
        print 
        rl.append(float(agent.learner.baseline))
    if save:
        fnStart="dataCartMom"
        fnExp=(repr(int(agent.learner.gd.alpha*100))+"m"+repr(int(agent.learner.gdSig.alpha*100))
               +"s"+repr(batch/2)+"b"+repr(int(agent.learner.epsilon*10))+"e")
        fnIdent="PGPE"+repr(int(random.random()*1000000.0))
        filename=fnStart+fnExp+fnIdent+".dat"
        file = open(filename,"w")
        rlLen=len(rl)
        for i in range(rlLen):
            file.write(repr((i+1)*batch*prnts)+"\n")
            file.write(repr(rl[i])+"\n")
        file.close()       
