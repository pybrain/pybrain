__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.tasks.capturegame import CaptureGameTask, RelativeCaptureTask
from pybrain.structure.evolvables.cheaplycopiable import CheaplyCopiable
from pybrain.structure.networks.custom.capturegame import CaptureGameNetwork
from pybrain.rl.learners.search import Coevolution
from pybrain.rl.agents.capturegameplayers import ModuleDecidingPlayer

size = 5
hsize = 1
popsize = 3

# total games to be played:
evals = 18

net = CaptureGameNetwork(size = size, hsize = hsize, simpleborders = True, #componentclass = MDLSTMLayer
                         )
net._params /= 20
net = CheaplyCopiable(net)

relativeTask = RelativeCaptureTask(size, useNetworks = True, maxGames = 10)

learner = Coevolution(relativeTask, [net], populationSize = popsize, verbose = True)

def main():
    newnet = learner.learn(evals)


from pybrain.tests.helpers import sortedProfiling
sortedProfiling('main()')  