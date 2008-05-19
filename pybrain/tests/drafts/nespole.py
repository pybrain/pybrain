__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.learners import NaturalEvolutionStrategies
from pybrain.rl.tasks.polebalancing import CartPoleTask
from pybrain import buildNetwork, FullConnection
from pybrain.tools.rankingfunctions import SmoothGiniRanking


def testNES():
    task = CartPoleTask(numPoles = 1, markov = True)
    net = buildNetwork(task.outdim, 1, task.indim)
    net.addRecurrentConnection(FullConnection(net['hidden0'], net['hidden0'], name = 'rec'))
    net.sortModules()
    l = NaturalEvolutionStrategies(task, net,
                                   lr = 0.5,
                                   rankingFunction = SmoothGiniRanking(#linearComponent = 0.2, 
                                                                       gini = 0.9,
                                                                       ),
                                   lambd = 100, 
                                   verbose = True,
                                   #veryverbose = True,
                                   )
    print l.learn()
        

if __name__ == '__main__':
    testNES()
