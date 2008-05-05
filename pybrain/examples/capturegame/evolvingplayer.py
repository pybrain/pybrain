""" A script illustrating how to evolve a simple Capture-Game Player
which uses a MDRNN as network, with a simple ES algorithm."""

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.tasks.capturegame import CaptureGameTask
from pybrain.structure.evolvables.cheaplycopiable import CheaplyCopiable
from pybrain.rl.learners import ES
from pybrain.utilities import storeCallResults
from pybrain.rl.agents.capturegameplayers.killing import KillingPlayer


# task settings: opponent, averaging to reduce noise, board size, etc.
size = 5
task = CaptureGameTask(size, averageOverGames = 40, 
                       opponent = KillingPlayer, 
                       opponentStart = False)

# keep track of evaluations for plotting
res = storeCallResults(task)

if False:
    # simple network
    from pybrain import buildNetwork, SigmoidLayer
    net = buildNetwork(task.outdim, task.indim, outclass = SigmoidLayer)
else:
    # specialized mdrnn variation
    from pybrain.structure.networks.custom.capturegame import CaptureGameNetwork
    net = CaptureGameNetwork(size = size)
    
net = CheaplyCopiable(net)
print net.name, 'has', net.paramdim, 'trainable parameters.'

learner = ES(task, net, mu = 5, lambada = 5, verbose = True, noisy = True)
learner.learn(500)

# plot the progression
import pylab
pylab.plot(res)
pylab.show()