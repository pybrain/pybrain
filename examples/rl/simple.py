__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'


""" This example demonstrates how to use the continuous Reinforcement Learning
    algorithms (ENAC, Reinforce) in a simple minimization task. 

    SimpleEnvironment is a one-dimensional quadratic function with its
    maximum at x=0. Additionally, noise can be added by setNoise(variance).
    The Agent can make steps in either direction and receives reward 
    equal to the negative function value. The optimal parameter is -10.0
   
    Requirements: pylab (for plotting only). If not available, 
    change the 'plotting' flag to False.
"""

from scipy import array, mean, zeros

from pybrain.rl.environments.simple import SimpleEnvironment, MinimizeTask 
from pybrain.tools.shortcuts import buildNetwork
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Reinforce
from pybrain.rl.experiments import EpisodicExperiment

# for plotting
plotting = True

if plotting:
    from pylab import draw, ion, title, plot, figure, clf
    ion()   

# create environment
env = SimpleEnvironment()
env.setNoise(0.1)

# create task
task = MinimizeTask(env)

# create controller network (no hidden layer, no bias) and initialize
net = buildNetwork(1, 1, bias=False, outputbias=False)
net._setParameters(array([-9.]))

# create agent with controller and learner
agent = LearningAgent(net, Reinforce())
agent.learner.learningRate = 0.01
agent.learner.gd.momentum = 0.9
# experiment
experiment = EpisodicExperiment(task, agent)

plots = zeros((1000, agent.module.paramdim+1), float)

for updates in range(1000):
    # training step
    experiment.doEpisodes(20)
    agent.learn()
    print "parameters:", agent.module.params
    print "sigma:", agent.learner.explorer.params
    # append mean reward to sr array
    ret = []
    for n in range(agent.history.getNumSequences()-1):
        state, action, reward = agent.history.getSequence(n)
        ret.append( sum(reward, 0).item() )

    plots[updates, 0] = mean(ret)
    plots[updates, 1:] = agent.module.params

    print "\nmean return %f\n" % (mean(ret))    
    if plotting and updates > 1:
        figure(1)
        clf()
        title('Reward (averaged over '+str(agent.history.getNumSequences())+' seqs)')
        plot(plots[0:updates, 0])
        figure(2)
        clf()
        title('Weight value')
        plot(plots[0:updates, 1])
        draw()
    agent.reset()
    
