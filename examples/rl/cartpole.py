__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.tools.shortcuts import buildNetwork
from pybrain.rl.environments.cartpole import CartPoleEnvironment, BalanceTask, CartPoleRenderer, EasyBalanceTask
from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import EpisodicExperiment
from pybrain.rl.learners import Reinforce, ENAC


episodes = 20
episodelength = 100

# create environment
env = CartPoleEnvironment()   
renderer = CartPoleRenderer()
env.setRenderer(renderer)

# create task
task = EasyBalanceTask(env, episodelength)

# create controller network
net = buildNetwork(4, 1, bias=False)

# create learner and agent
learner = Reinforce()
learner.learningRate = 0.1
agent = LearningAgent(net, learner)

# create experiment
renderer.start()
env.delay = False
experiment = EpisodicExperiment(task, agent)

for i in range(100):
    env.delay = True
    experiment.doEpisodes(1)
    env.delay = False
    experiment.doEpisodes(episodes)
    print "learn"
    agent.learn()
    agent.reset()




                         
