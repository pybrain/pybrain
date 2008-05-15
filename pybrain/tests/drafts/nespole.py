__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.learners import NaturalEvolutionStrategies
from pybrain.rl.tasks.polebalancing import CartPoleTask
from pybrain import buildNetwork, FullConnection


def testNES():
    task = CartPoleTask(numPoles = 2, markov = False)
    net = buildNetwork(task.outdim, 2, task.indim)
    net.addRecurrentConnection(FullConnection(net['hidden0'], net['hidden0'], name = 'rec'))
    net.sortModules()
    l = NaturalEvolutionStrategies(task, net,
                                   lr = 0.002,
                                   lambd = 250, 
                                   gini=0.1,
                                   ranking = 'smooth', 
                                   verbose = True,
                                   )
    print l.learn()
        

if __name__ == '__main__':
    testNES()
