#########################################################################
# Reinforcement Learning with PGPE on the ShipSteering Environment
#
# Requirements: 
#   pybrain (tested on rev. 1195, ship env rev. 1202)
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

batch=2
prnts=50
epis=2000/batch/prnts
numbExp=10
et = ExTools(batch, prnts)

for runs in range(numbExp):
    # create environment
    #Options: Bool(OpenGL), Bool(Realtime simu. while client is connected), ServerIP(default:localhost), Port(default:21560)
    env = ShipSteeringEnvironment(False)
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
    
    for updates in range(epis):
        for i in range(prnts):
            experiment.doEpisodes(batch)
        et.printResults((agent.learner._allEvaluations)[-50:-1], runs, updates)
    et.addExps()
et.showExps()
