#########################################################################
# Reinforcement Learning with PGPE/SPLA on the ShipSteering Environment
#
# Requirements: 
#   pybrain (tested on rev. 1195, ship env rev. 1202)
# Author: Frank Sehnke, sehnke@in.tum.de
#########################################################################
__author__ = "Martin Felder, Frank Sehnke"
__version__ = '$Id$' 

#--- 
# default backend GtkAgg does not plot properly on Ubuntu 8.04
import matplotlib
matplotlib.use('TkAgg')
#---

from pybrain.structure.modules.tanhlayer import TanhLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.rl.environments.shipsteer import ShipSteeringEnvironment
from pybrain.rl.environments.shipsteer import GoNorthwardTask
from pybrain.rl.agents.finitedifference import FiniteDifferenceAgent
from pybrain.optimization import PGPE as SPLA
from pybrain.rl.experiments import EpisodicExperiment
from pybrain.tools.plotting import MultilinePlotter
from pylab import figure, ion #@UnresolvedImport
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

useGraphics = False
if useGraphics:
    figure()
    ion()
    pl = MultilinePlotter(autoscale=1.2, xlim=[0, 50], ylim=[0, 1])
    pl.setLineStyle(linewidth=2)

numbExp=25 #number of experiments
for runs in range(numbExp):
    # create environment
    #Options: Bool(OpenGL), Bool(Realtime simu. while client is connected), ServerIP(default:localhost), Port(default:21560)
    env = ShipSteeringEnvironment(False)
    # create task
    task = GoNorthwardTask(env,maxsteps = 500)
    # create controller network
    net = buildNetwork(task.outdim, task.indim, outclass=TanhLayer)
    # create agent with controller and learner
    agent = FiniteDifferenceAgent(net, SPLA())
    # learning options
    agent.learner.gd.alpha = 0.3 #step size of \mu adaption
    agent.learner.gdSig.alpha = 0.15 #step size of \sigma adaption
    agent.learner.gd.momentum = 0.0
    batch=2 #number of samples per gradient estimate (was: 2; more here due to stochastic setting)
    #create experiment
    experiment = EpisodicExperiment(task, agent)
    prnts=1 #frequency of console output
    epis=2000/batch/prnts
    
    #actual roll outs
    filename="dataSPLA08NoRew"+repr(int(random.random()*1000000.0))+".dat"
    wf = open(filename, 'wb')
    for updates in range(epis):
        for i in range(prnts):
            experiment.doEpisodes(batch) #execute #batch episodes
            agent.learn() #learn from the gather experience
            agent.reset() #reset agent and environment
        #print out related data
        stp = (updates+1)*batch*prnts
        print "Step: ", runs, "/", stp, "Best: ", agent.learner.best, "Base: ", agent.learner.baseline, "Reward: ", agent.learner.reward   
        wf.write(repr(stp)+"\n") 
        wf.write(repr(agent.learner.baseline[0])+"\n") 
        if useGraphics:
            pl.addData(0,float(stp),agent.learner.baseline)
            pl.addData(1,float(stp),agent.learner.best)
            pl.update()

        #if updates/100 == float(updates)/100.0:
        #    saveWeights("walk.wgt", agent.learner.original)  
    wf.close()      
