###########################################################################
# This program takes 4 parameters at the command line and runs the
# (single) cartpole environment with it, visualizing the cart and the pole.
# if cart is green, no penality is given. if the cart is blue, a penalty of
# -1 per step is given. the program ends with the end of the episode. if
# the variable "episodes" is changed to a bigger number, the task is executed
# faster and the mean return of all episodes is printed.
###########################################################################
__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain import *
from pybrain.tools.shortcuts import *
from pybrain.rl.environments.cartpole import *
from pybrain.rl.agents.learning import LearningAgent
from pybrain.rl.learners import *
from pybrain.rl.experiments import EpisodicExperiment
from scipy import array, mean
import os, sys

episodes = 100
epilen = 200

if len(sys.argv) < 5:
    sys.exit('please give 4 parameters. run: "python play.py <p1> <p2> <p3> <p4>"\n')
     
# create environment
env = CartPoleEnvironment()    
env.setRenderer(CartPoleRenderer())
env.getRenderer().start()
env.delay = (episodes == 1)

# create task
task = BalanceTask(env, epilen)
# create controller network
net = buildNetwork(4, 1, bias=False)
# set parameters from command line
# create agent
agent = LearningAgent(net, None)
agent.module._setParameters(array([float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])])) 
agent.disableLearning()
# create experiment
experiment = EpisodicExperiment(task, agent)
experiment.doEpisodes(episodes)
ret = []
for n in range(agent.history.getNumSequences()):
    returns = agent.history.getSequence(n)
    reward = returns[2]
    ret.append( sum(reward, 0).item() )
print ret, "mean:",mean(ret)
env.getRenderer().stop()

                         
