#########################################################################
# Reinforcement Learning with Policy Gradients on the SimpleEnvironment 
#
# SimpleEnvironment is a one-dimensional quadratic function with its
# maximum at x=0. Additionally, noise can be added (setNoise(variance)).
# The Agent can make steps in either direction and has receives reward 
# equal to the negative function value. The optimal parameter is -10.0

# Requirements: pylab (for plotting only). If not available, 
# change the 'plotting' flag to False
#########################################################################

from pybrain import buildNetwork
from pybrain.rl.environments.simple import SimpleEnvironment, MinimizeTask
from pybrain.rl.agents import PolicyGradientAgent
from pybrain.rl.learners import ENAC
from pybrain.rl.experiments import EpisodicExperiment
from scipy import array, mean, zeros


# for plotting
plotting = True


if plotting:
    from pylab import draw, ion, title, plot, figure, clf #@UnresolvedImport
    ion()   

# create environment
env = SimpleEnvironment()
env.setNoise(0.9)
# create task
task = MinimizeTask(env)
# create controller network (flat network)
net = buildNetwork(1, 1, bias=False)
net._setParameters(array([-11.]))
# create agent with controller and learner
agent = PolicyGradientAgent(net, ENAC())
# initialize parameters (variance)
agent.setSigma([-2.])
# learning options
agent.learner.alpha = 2.
# agent.learner.rprop = True
agent.actaspg = False
experiment = EpisodicExperiment(task, agent)


plots = zeros((1000, agent.module.paramdim+1), float)

for updates in range(1000):
    agent.reset()
    # training step
    experiment.doEpisodes(10)
    agent.learn()
    print "parameters:", agent.module.params

    # append mean reward to sr array
    ret = []
    for n in range(agent.history.getNumSequences()):
        state, action, reward, _ = agent.history.getSequence(n)
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
