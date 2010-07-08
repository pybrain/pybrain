__author__ = 'Tom Schaul, tom@idsia.ch'

from random import choice

from randomplayer import RandomCapturePlayer


class NonSuicidePlayer(RandomCapturePlayer):
    """ do random non-suicide moves in the capture game """
    def getAction(self):
        p = self.game.getAcceptable(self.color)
        if len(p) > 0:
            return [self.color, choice(p)]
        else:
            return RandomCapturePlayer.getAction(self)