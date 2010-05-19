__author__ = 'Tom Schaul, tom@idsia.ch'

from random import choice

from randomplayer import RandomGomokuPlayer


class KillingGomokuPlayer(RandomGomokuPlayer):
    """ do random moves, but always instant-kill if possible. """
    def getAction(self):
        p = self.game.getKilling(self.color)
        if len(p) > 0:
            return [self.color, choice(p)]
        else:
            return RandomGomokuPlayer.getAction(self)