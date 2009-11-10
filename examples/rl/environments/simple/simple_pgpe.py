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
from pybrain.rl.environments.simple import SimpleEnvironment, MinimizeTask
from pybrain.rl.agents import OptimizationAgent
from pybrain.optimization import PGPE
from pybrain.rl.experiments import EpisodicExperiment
from scipy import random

numbExp=40
for runs in range(numbExp):
    # create environment
    env = SimpleEnvironment()
    env.setNoise(0.01)
    # create task
    task = MinimizeTask(env)
    # create controller network (flat network)
    net = buildNetwork(1, 1, bias=False)
    # create agent with controller and learner (and its options)
    agent = OptimizationAgent(net, PGPE(learningRate = 0.3,
                                    sigmaLearningRate = 0.15,
                                    momentum = 0.0,
                                    epsilon = 2.0,
                                    #rprop = True,
                                    ))

    experiment = EpisodicExperiment(task, agent)
    batch=2 #with PGPE this must be a even number (2 for all deterministic settings)
    prnts=1 
    epis=200/batch/prnts
    save=True

    rl=[]
    pr=[]
    for updates in range(epis):
        for i in range(prnts):
            experiment.doEpisodes(batch)
            #agent.learn()
            #agent.reset()
        print "Parameters:", agent.learner.original, "Sigmas:", agent.learner.sigList 
        print "Episode:", runs, "/", (updates+1)*prnts*batch, "Best:", agent.learner.bestEvaluation, "Base:", agent.learner.baseline, "Reward:", agent.learner.mreward
        print ""    
        rl.append(float(agent.learner.baseline))
        pr.append(float(agent.learner.original[0]))

    if save:
        fnStart="dataSimple"
        fnExp=repr(int(agent.learner.gd.alpha*100))+"m"+repr(int(agent.learner.gdSig.alpha*100))+"s"+repr(batch/2)+"b"+repr(int(agent.learner.epsilon*10))+"e"
        fnIdent="PGPE"+repr(int(random.random()*1000000.0))
        filename=fnStart+fnExp+fnIdent+".dat"
        file = open(filename,"w")
        rlLen=len(rl)
        for i in range(rlLen):
            file.write(repr((i+1)*batch*prnts)+"\n")
            file.write(repr(rl[i])+"\n")
        file.close()       
     
        fnStart="dataSimplePara"
        fnExp=repr(int(agent.learner.gd.alpha*100))+"m"+repr(int(agent.learner.gdSig.alpha*100))+"s"+repr(batch/2)+"b"+repr(int(agent.learner.epsilon*10))+"e"
        fnIdent="PGPE"+repr(int(random.random()*1000000.0))
        filename=fnStart+fnExp+fnIdent+".dat"
        file = open(filename,"w")
        rlLen=len(pr)
        for i in range(rlLen):
            file.write(repr((i+1)*batch*prnts)+"\n")
            file.write(repr(pr[i])+"\n")
        file.close()            
