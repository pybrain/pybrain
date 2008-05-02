""" A script illustrating how to evolve a simple Capture-Game Player
which uses a MDRNN as network, with a simple ES algorithm."""

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.tasks.capturegame import CaptureGameTask
from pybrain.structure.networks import BorderSwipingNetwork
from pybrain.structure.evolvables.cheaplycopiable import CheaplyCopiable
from pybrain.rl.learners import ES
from pybrain import buildNetwork, SigmoidLayer
from pybrain.utilities import storeCallResults

# default settings: random opponent, size = 5, value = average over 10 games.
task = CaptureGameTask(5, averageOverGames = 100, opponentStart = False)

# keep track of evaluations for plotting
res = storeCallResults(task)


# TODO: build the MDRNN:
#net = BorderSwipingNetwork(inmesh, hiddenmesh, outmesh)

net = buildNetwork(task.outdim, task.indim, outclass = SigmoidLayer)
net = CheaplyCopiable(net)


learner = ES(task, net, mu = 5, lambada = 5, verbose = True, noisy = True)
learner.learn(500)

# plot the progression
import pylab
pylab.plot(res)
pylab.show()