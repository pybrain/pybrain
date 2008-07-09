""" A little example script showing a Capture-Game tournament between 
 - a random player
 - a kill-on-sight player
 - a small-network-player with random weights
"""

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.environments.twoplayergames import CaptureGame
from pybrain.rl.agents.capturegameplayers import RandomCapturePlayer, KillingPlayer, ModuleDecidingPlayer
from pybrain.rl.agents.capturegameplayers.clientwrapper import ClientCapturePlayer
from pybrain.rl.experiments import Tournament
from pybrain import buildNetwork, SigmoidLayer

game = CaptureGame(5)
randAgent = RandomCapturePlayer(game, name = 'rand')
killAgent = KillingPlayer(game, name = 'kill')
javaAgent = ClientCapturePlayer(game, name = 'java')

# the network's outputs are probabilities of choosing the action, thus a sigmoid output layer
net = buildNetwork(game.outdim, game.indim, outclass = SigmoidLayer)
netAgent = ModuleDecidingPlayer(net, game, name = 'net')

# same network, but greedy decisions:
netAgentGreedy = ModuleDecidingPlayer(net, game, name = 'greedy', greedySelection = True)

tourn = Tournament(game, [randAgent, killAgent, netAgent, netAgentGreedy, javaAgent])
tourn.organize(50)
print tourn

# try a different network, and play again:
net.randomize()
tourn.reset()
tourn.organize(50)
print tourn



