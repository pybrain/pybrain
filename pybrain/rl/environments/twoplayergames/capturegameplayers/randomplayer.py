__author__ = 'Tom Schaul, tom@idsia.ch'

from random import choice

from captureplayer import CapturePlayer


class RandomCapturePlayer(CapturePlayer):
    """ do random moves in the capture game"""

    def getAction(self):
        return [self.color, choice(self.game.getLegals(self.color))]