#########################################################################
# Reinforcement Learning with SimpleSPSA on the ShipSteering Environment
#
# Requirements: 
#   pybrain (tested on rev. 1195, ship env rev. 1202)
#########################################################################
__author__ = "Martin Felder, Frank Sehnke"
__version__ = '$Id: shipbenchSPLA.py 1305 2008-06-10 11:51:18Z sehnke $' 

from pybrain.structure.modules.tanhlayer import TanhLayer
from pybrain import buildNetwork
from pybrain.rl.environments.shipsteer import ShipSteeringEnvironment
from pybrain.rl.environments.shipsteer import GoNorthwardTask
from pybrain.rl.agents.finitedifference import FiniteDifferenceAgent
from pybrain.rl.learners.finitedifference.spsa import SimpleSPSA
from pybrain.rl.experiments import EpisodicExperiment
from cPickle import load, dump
from scipy import random

# Method for loading a weight matrix and initialize the network
def loadWeights(filename):
    filepointer = file(filename)
    agent.learner.original = load(filepointer)
    filepointer.close()
    agent.learner.gd.init(agent.learner.original)

# Method for saving the weight matrix    
def saveWeights(filename, w):
    filepointer = file(filename, 'w+')
    dump(w, filepointer)
    filepointer.close()

numbExp=1 #number of experiments
for runs in range(numbExp):
    # create environment
    #Options: Bool(OpenGL), Bool(Realtime simu. while client is connected), ServerIP(default:localhost), Port(default:21560)
    env = ShipSteeringEnvironment(False)
    # create task
    task = GoNorthwardTask(env,maxsteps = 500)
    # create controller network
    net = buildNetwork(task.outdim, task.indim, outclass=TanhLayer)
    # create agent with controller and learner
    agent = FiniteDifferenceAgent(net, SimpleSPSA())
    # learning options
    agent.learner.gd.alpha = 0.5 #step size of parameter adaption
    agent.learner.gamma=0.9993 #exploration decay factor
    agent.learner.gd.momentum = 0.0
    batch=2 #number of samples per gradient estimate (Symetric sampling needs odd number of samples per estimate!)
    #create experiment
    experiment = EpisodicExperiment(task, agent)
    prnts=10 #frequency of console output
    epis=10000/batch/prnts
    
    #actual roll outs
    #filename="dataSPLA08NoRew"+repr(int(random.random()*1000000.0))+".dat"
    #wf = open(filename, 'wb')
    for updates in range(epis):
        for i in range(prnts):
            experiment.doEpisodes(batch) #execute #batch episodes
            agent.learn() #learn from the gather experience
            agent.reset() #reset agent and environment
        #print out related data
        stp = (updates+1)*batch*prnts
        print "AverParam:", abs(agent.learner.original).sum()/agent.learner.numOParas, "Exploration:", agent.learner.epsilon
        print "Step: ", runs, "/", stp, "Best: ", agent.learner.best, "Base: ", agent.learner.baseline, "Reward: ", agent.learner.reward   
        #wf.write(repr(stp)+"\n") 
        #wf.write(repr(agent.learner.baseline[0])+"\n") 

        #if updates/100 == float(updates)/100.0:
        #    saveWeights("walk.wgt", agent.learner.original)  
    #wf.close()      
