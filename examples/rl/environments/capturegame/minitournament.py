from __future__ import print_function

#!/usr/bin/env python
""" A little example script showing a Capture-Game tournament between
 - a random player
 - a kill-on-sight player
 - a small-network-player with random weights
"""

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.environments.twoplayergames import CaptureGame
from pybrain.rl.environments.twoplayergames.capturegameplayers import RandomCapturePlayer, KillingPlayer, ModuleDecidingPlayer
from pybrain.rl.environments.twoplayergames.capturegameplayers.clientwrapper import ClientCapturePlayer
from pybrain.rl.experiments.tournament import Tournament
from pybrain.tools.shortcuts import buildNetwork
from pybrain import SigmoidLayer

game = CaptureGame(5)
randAgent = RandomCapturePlayer(game, name = 'rand')
killAgent = KillingPlayer(game, name = 'kill')

# the network's outputs are probabilities of choosing the action, thus a sigmoid output layer
net = buildNetwork(game.outdim, game.indim, outclass = SigmoidLayer)
netAgent = ModuleDecidingPlayer(net, game, name = 'net')

# same network, but greedy decisions:
netAgentGreedy = ModuleDecidingPlayer(net, game, name = 'greedy', greedySelection = True)

agents = [randAgent, killAgent, netAgent, netAgentGreedy]

try:
    javaAgent = ClientCapturePlayer(game, name = 'java')
    agents.append(javaAgent)
except:
    print('No Java server available.')

print()
print('Starting tournament...')
tourn = Tournament(game, agents)
tourn.organize(50)
print(tourn)

# try a different network, and play again:
net.randomize()
tourn.reset()
tourn.organize(50)
print(tourn)



