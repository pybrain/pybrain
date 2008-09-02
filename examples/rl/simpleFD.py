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

from pybrain.tools.shortcuts import buildNetwork
from pybrain.rl.environments.simple import SimpleEnvironment, MinimizeTask
from pybrain.rl.agents import FiniteDifferenceAgent
from pybrain.rl.learners import FDBasic
from pybrain.rl.experiments import EpisodicExperiment
from scipy import array, mean

from pylab import ion

# for plotting
ion()   

# create environment
env = SimpleEnvironment()
env.setNoise(0.01)
# create task
task = MinimizeTask(env)
# create controller network (flat network)
net = buildNetwork(1, 1, bias=False)
net._setParameters(array([0.0]))
# create agent with controller and learner
agent = FiniteDifferenceAgent(net, FDBasic())
# initialize parameters (variance)
#agent.setSigma([-2.])
# learning options
agent.learner.alpha = 0.1
# agent.learner.rprop = True
experiment = EpisodicExperiment(task, agent)

best=0.0
base=0.0
rew=0.0

for updates in range(1000):

    # testing step
    agent.disableLearning()
    experiment.doEpisodes(2)
    
    # append mean reward to sr array
    ret = []
    for n in range(agent.history.getNumSequences()):
        state, action, reward = agent.history.getSequence(n)
        ret.append( sum(reward, 0).item() )
    rew=mean(ret)
    base=0.9*base+0.1*rew
    if rew>best: best=rew
    print "Parameters:", agent.module.params, "Epsilon: ", agent.learner.epsilon, "Best: ", best, "Base: ", base, "Reward %f\n" % rew    
    agent.enableLearning()
    agent.reset()

    # training step
    for i in range(5):
        experiment.doEpisodes(10)
        agent.learn()
        agent.reset()
            
