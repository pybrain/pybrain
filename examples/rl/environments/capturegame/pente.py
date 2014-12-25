from __future__ import print_function

#!/usr/bin/env python
""" A little script illustrating how to use a (randomly initialized)
convolutional network to play a game of Pente. """

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.environments.twoplayergames.pente import PenteGame
from pybrain.rl.environments.twoplayergames.gomokuplayers.randomplayer import RandomGomokuPlayer
from pybrain.rl.environments.twoplayergames.gomokuplayers.moduledecision import ModuleDecidingPlayer
from pybrain.structure.networks.custom.convboard import ConvolutionalBoardNetwork

dim = 7
g = PenteGame((dim, dim))
print(g)
n = ConvolutionalBoardNetwork(dim, 5, 3)
p1 = ModuleDecidingPlayer(n, g)
p2 = RandomGomokuPlayer(g)
p2.color = g.WHITE
g.playToTheEnd(p1, p2)
print(g)
