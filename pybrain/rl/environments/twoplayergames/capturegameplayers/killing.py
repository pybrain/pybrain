__author__ = 'Tom Schaul, tom@idsia.ch'

from random import choice

from nonsuicide import NonSuicidePlayer


class KillingPlayer(NonSuicidePlayer):
    """ do random moves, but always instant-kill if possible,
    and never suicide, if avoidable. """
    def getAction(self):
        p = self.game.getKilling(self.color)
        if len(p) > 0:
            return [self.color, choice(p)]
        else:
            return NonSuicidePlayer.getAction(self)