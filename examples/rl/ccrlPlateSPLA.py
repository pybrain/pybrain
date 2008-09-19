#########################################################################
# Reinforcement Learning with PGPE/SPLA on the CCRL ODE Environment 
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
# Requirements: scipy for the environment and the learner; ODE.
#
# Author: Frank Sehnke, sehnke@in.tum.de
#########################################################################

from pybrain.rl.environments.ode import CCRLEnvironment
from pybrain.rl.environments.ode.tasks import CCRLPlateTask
from pybrain.structure.modules.tanhlayer import TanhLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.rl.agents.finitedifference import FiniteDifferenceAgent
from pybrain.rl.learners.finitedifference.spla import SPLA
from pybrain.rl.experiments import EpisodicExperiment
from cPickle import load, dump

# Method for loading a weight matrix and initialize the network
def loadWeights(filename):
    filepointer = file(filename)
    original = load(filepointer)
    filepointer.close()
    return original

# Method for saving the weight matrix    
def saveWeights(filename, w):
    filepointer = file(filename, 'w+')
    dump(w, filepointer)
    filepointer.close()

hiddenUnits = 10
loadNet=False
saveNet=False
saveName="plate.wgt"
numbExp=1 #number of experiments
for runs in range(numbExp):
    # create environment
    #Options: XML-Model, Bool(OpenGL), Bool(Realtime simu. while client is connected), ServerIP(default:localhost), Port(default:21560)
    env = CCRLEnvironment("ccrlPlate.xode")
    # create task
    task = CCRLPlateTask(env)
    # create controller network
    net = buildNetwork(len(task.getObservation()), hiddenUnits, env.actLen, outclass=TanhLayer)    
    # create agent with controller and learner
    agent = FiniteDifferenceAgent(net, SPLA())
    # learning options
    agent.learner.gd.alpha = 0.2 #step size of \mu adaption
    agent.learner.gdSig.alpha = 0.085 #step size of \sigma adaption
    agent.learner.gd.momentum = 0.0
    
    #Loading weights
    if loadNet:
        agent.learner.original=loadWeights("plate.wgt")
        agent.learner.gd.init(agent.learner.original)
        agent.learner.epsilon=0.2
        agent.learner.initSigmas()

    batch=2 #number of samples per gradient estimate
    #create experiment
    experiment = EpisodicExperiment(task, agent)
    prnts=1 #frequency of console output
    epis=5000000/batch/prnts
    
    #actual roll outs
    for updates in range(epis):
        for i in range(prnts):
            experiment.doEpisodes(batch) #execute batch episodes
            agent.learn() #learn from the gather experience
            agent.reset() #reset agent and environment
        #print out related data
        print "Step: ", runs, "/", (updates+1)*batch*prnts, "Best: ", agent.learner.best, 
        print "Base: ", agent.learner.baseline, "Reward: ", agent.learner.reward 
        #Saving weights
        if saveNet:
            if updates/100 == float(updates)/100.0: saveWeights(saveName, agent.learner.original)  
