#########################################################################
# Reinforcement Learning with Policy Gradients on the SimpleEnvironment 
#
# SimpleEnvironment is a one-dimensional quadratic function with its
# maximum at x=0. Additionally, noise can be added (setNoise(variance)).
# The Agent can make steps in either direction and has receives reward 
# equal to the negative function value. The optimal parameter is -10.0

# Requirements: pylab (for plotting only). If not available, comment the
# 9 lines total marked as "for plotting"
#########################################################################

from pybrain import *
from pybrain.tools.shortcuts import *
from pybrain.rl.environments.ode import *
from pybrain.rl.environments.ode.tasks import *

from pybrain.rl.agents import FiniteDifferenceAgent
from pybrain.rl.learners import *
from pybrain.rl.experiments import EpisodicExperiment
from scipy import array, mean, random
from cPickle import load, dump

def loadWeights(filename):
    filepointer = file(filename)
    org = load(filepointer)
    filepointer.close()
    return org

def saveWeights(filename, w):
    filepointer = file(filename, 'w+')
    dump(w, filepointer)
    filepointer.close()

numbExp=1
for runs in range(numbExp):
    # create environment
    env = JohnnieEnvironment(True) #True for OpenGL output
    # create task
    task = StandingTask(env) #Other tasks available are StandingTask and JumpingTask
    # create controller network
    net = buildNetwork(env.obsLen, 10, env.actLen, outclass=TanhLayer)
    # create agent with controller and learner
    agent = FiniteDifferenceAgent(net, SPLA())
    # learning options
    agent.learner.gd.alpha = 0.2
    agent.learner.gdSig.alpha = 0.1
    agent.learner.gd.momentum = 0.0
    agent.learner.epsilon = 2.0
    agent.learner.initSigmas()
    # agent.learner.rprop = True
    experiment = EpisodicExperiment(task, agent)
    batch=2
    prnts=1
    epis=5000/batch/prnts
    if env.hasRenderer(): 
        #set lower framerate for computers with bad grafic chip
        env.getRenderer().setFrameRate(2) 
        env.getRenderer().start()      
    
    rl=[]
    for updates in range(epis):
        for i in range(prnts):
            experiment.doEpisodes(batch)
            agent.learn()
            agent.reset()
        print "Step: ", runs, "/", (updates+1)*batch*prnts, "Best: ", agent.learner.best, "Base: ", agent.learner.baseline, "Reward: ", agent.learner.reward   
        print ""
        rl.append(float(agent.learner.baseline))
        #if updates/100 == float(updates)/100.0:
        #    saveWeights("standUp.wgt", agent.learner.original)     
