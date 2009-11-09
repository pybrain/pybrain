__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'


from pybrain.rl.environments.cartpole import CartPoleEnvironment, DiscreteBalanceTask, CartPoleRenderer
from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import EpisodicExperiment
from pybrain.rl.learners.valuebased import NFQ, ActionValueNetwork
from pybrain.rl.explorers import BoltzmannExplorer

from numpy import array, arange, meshgrid, pi, zeros, mean
from matplotlib import pyplot as plt

plt.ion()

env = CartPoleEnvironment()
renderer = CartPoleRenderer()
env.setRenderer(renderer)
# renderer.start()

module = ActionValueNetwork(2, 3)
task = DiscreteBalanceTask(env, 50)
learner = NFQ()
learner.explorer = BoltzmannExplorer()
agent = LearningAgent(module, learner)
testagent = LearningAgent(module, None)
experiment = EpisodicExperiment(task, agent)

def plotStateValues(module, fig):
    plt.figure(fig.number)
    theta_ = arange(-pi/2, pi/2, 0.1)
    v_ = arange(-5, 5, 0.3)
    
    X,Y = meshgrid(theta_, v_)
    X = X.flatten()
    Y = Y.flatten()
    
    Q = zeros(len(theta_) * len(v_))
    
    for i, (theta, v) in enumerate(zip(X, Y)):
        Q[i] = max(module.getActionValues(array([theta, v])))
    Q = Q.reshape(len(v_), len(theta_))
    
    plt.gray()
    plt.imshow(Q, interpolation='nearest')
    plt.axis('tight')
    plt.gcf().canvas.draw()

def plotPerformance(values, fig):
    plt.figure(fig.number)
    plt.clf()
    plt.plot(values, 'o-')
    plt.gcf().canvas.draw()


performance = []

sv_fig = plt.figure()
pf_fig = plt.figure()

# experiment.doEpisodes(50)
    
while(True):
    env.delay = True
    experiment.doEpisodes(10)
    env.delay = False

    while agent.history.getNumSequences() > 50:
        agent.history.removeSequence(0)
        
    agent.learn(20)
    
    experiment.agent = testagent
    r = mean([sum(x) for x in experiment.doEpisodes(20)])

    testagent.reset()
    experiment.agent = agent
    

    performance.append(r) 
    plotStateValues(module, sv_fig)
    plotPerformance(performance, pf_fig)
    print "reward avg", r
    print "params", agent.module.network.params
    # print "exploration", agent.learner.explorer.epsilon
    print "num samples", agent.history.getNumSequences()
    print "update step", len(performance)
    # agent.reset()
    
