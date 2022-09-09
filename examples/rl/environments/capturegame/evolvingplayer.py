from __future__ import print_function

#!/usr/bin/env python
""" A script illustrating how to evolve a simple Capture-Game Player
which uses a MDRNN as network, with a simple ES algorithm."""

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.environments.twoplayergames import CaptureGameTask
from pybrain.structure.evolvables.cheaplycopiable import CheaplyCopiable
from pybrain.optimization import ES
from pybrain.utilities import storeCallResults
from pybrain.rl.environments.twoplayergames.capturegameplayers.killing import KillingPlayer

# task settings: opponent, averaging to reduce noise, board size, etc.
size = 5
simplenet = False
task = CaptureGameTask(size, averageOverGames = 40, opponent = KillingPlayer)

# keep track of evaluations for plotting
res = storeCallResults(task)

if simplenet:
    # simple network
    from pybrain.tools.shortcuts import buildNetwork
    from pybrain import SigmoidLayer
    net = buildNetwork(task.outdim, task.indim, outclass = SigmoidLayer)
else:
    # specialized mdrnn variation
    from pybrain.structure.networks.custom.capturegame import CaptureGameNetwork
    net = CaptureGameNetwork(size = size, hsize = 2, simpleborders = True)

net = CheaplyCopiable(net)
print(net.name, 'has', net.paramdim, 'trainable parameters.')

learner = ES(task, net, mu = 5, lambada = 5,
             verbose = True, evaluatorIsNoisy = True,
             maxEvaluations = 50)
newnet, f = learner.learn()

# now, let's take the result, and compare it's performance on a larger game-baord (to the original one)
newsize = 7
bignew = newnet.getBase().resizedTo(newsize)
bigold = net.getBase().resizedTo(newsize)

print('The rescaled network,', bignew.name, ', has', bignew.paramdim, 'trainable parameters.')

newtask = CaptureGameTask(newsize, averageOverGames = 50, opponent = KillingPlayer)
print('Old net on big board score:', newtask(bigold))
print('New net on big board score:', newtask(bignew))


# plot the progression
from pylab import plot, show
plot(res)
show()