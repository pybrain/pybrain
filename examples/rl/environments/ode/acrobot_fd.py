#########################################################################
# Reinforcement Learning with PGPE on the FlexCube Environment 
#
# The FlexCube Environment is a Mass-Spring-System composed of 8 mass points.
# These resemble a cube with flexible edges.
#
# Control/Actions:
# The agent can control the 12 equilibrium edge lengths. 
#
# A wide variety of sensors are available for observation and reward:
# - 12 edge lengths
# - 12 wanted edge lengths (the last action)
# - vertexes contact with floor
# - vertexes min height (distance of closest vertex to the floor)
# - distance to origin
# - distance and angle to target
#
# Task available are:
# - GrowTask, agent has to maximize the volume of the cube
# - JumpTask, agent has to maximize the distance of the lowest mass point during the episode
# - WalkTask, agent has to maximize the distance to the starting point
# - WalkDirectionTask, agent has to minimize the distance to a target point.
# - TargetTask, like the previous task but with several target points
# 
# Requirements: scipy for the environment and the learner.
#
# Author: Frank Sehnke, sehnke@in.tum.de
#########################################################################

from pybrain.tools.example_tools import ExTools
from pybrain.rl.environments.ode import AcrobotEnvironment
from pybrain.rl.environments.ode.tasks import GradualRewardTask
from pybrain.structure.modules.tanhlayer import TanhLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.rl.agents import OptimizationAgent
from pybrain.optimization import FiniteDifferences
from pybrain.rl.experiments import EpisodicExperiment

batch=2
prnts=1
epis=4000/batch/prnts
numbExp=10
et = ExTools(batch, prnts)

for runs in range(numbExp):
    # create environment
    #Options: Bool(OpenGL), Bool(Realtime simu. while client is connected), ServerIP(default:localhost), Port(default:21560)
    env = AcrobotEnvironment() 
    # create task
    task = GradualRewardTask(env)
    # create controller network
    net = buildNetwork(len(task.getObservation()), env.actLen, outclass=TanhLayer)    
    # create agent with controller and learner (and its options)
    agent = OptimizationAgent(net, FiniteDifferences(storeAllEvaluations = True))
    et.agent = agent
    experiment = EpisodicExperiment(task, agent)

    for updates in range(epis):
        for i in range(prnts):
            experiment.doEpisodes(batch)
        et.printResults((agent.learner._allEvaluations)[-50:-1], runs, updates)
    et.addExps()
et.showExps()
