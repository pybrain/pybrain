#########################################################################
# Reinforcement Learning with PGPE on the CartPoleEnvironment 
#
# Requirements: pylab (for plotting only). If not available, comment the
# 9 lines total marked as "for plotting"
# Author: Frank Sehnke, sehnke@in.tum.de
#########################################################################

from pybrain.tools.example_tools import ExTools
from pybrain.tools.shortcuts import buildNetwork
from pybrain.rl.environments.cartpole import CartPoleEnvironment, BalanceTask
from pybrain.rl.agents import OptimizationAgent
from pybrain.optimization import ExactNES
from pybrain.rl.experiments import EpisodicExperiment

batch=2
prnts=100
epis=4000/batch/prnts
numbExp=10
et = ExTools(batch, prnts)

for runs in range(numbExp):
    env = CartPoleEnvironment()    
    # create task
    task = BalanceTask(env, 200, desiredValue=None)
    # create controller network
    net = buildNetwork(4, 1, bias=False)
    # create agent with controller and learner (and its options)
    agent = OptimizationAgent(net, ExactNES(storeAllEvaluations = True))
    et.agent = agent
    experiment = EpisodicExperiment(task, agent)
    
    for updates in range(epis):
        for i in range(prnts):
            experiment.doEpisodes(batch)
        print "Epsilon   : ", agent.learner.sigma
        et.printResults((agent.learner._allEvaluations)[-50:-1], runs, updates)
    et.addExps()
et.showExps()
