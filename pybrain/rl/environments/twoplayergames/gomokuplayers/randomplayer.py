__author__ = 'Tom Schaul, tom@idsia.ch'

from random import choice

from gomokuplayer import GomokuPlayer


class RandomGomokuPlayer(GomokuPlayer):
    """ do random moves in Go-Moku"""

    def getAction(self):
        return [self.color, choice(self.game.getLegals(self.color))]