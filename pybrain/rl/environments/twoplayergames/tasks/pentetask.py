
__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.environments.twoplayergames.pente import PenteGame
from pybrain.rl.environments.episodic import EpisodicTask
from gomokutask import GomokuTask
from pybrain.rl.environments.twoplayergames.gomokuplayers import RandomGomokuPlayer
from inspect import isclass


class PenteTask(GomokuTask):
    """ The task of winning the maximal number of Gomoku games against a fixed opponent. """

    def __init__(self, size, opponent = None, **args):
        EpisodicTask.__init__(self, PenteGame((size, size)))
        self.setArgs(**args)
        if opponent == None:
            opponent = RandomGomokuPlayer(self.env)
        elif isclass(opponent):
            # assume the agent can be initialized without arguments then.
            opponent = opponent(self.env)
        if not self.opponentStart:
            opponent.color = PenteGame.WHITE
        self.opponent = opponent
        self.minmoves = 9
        self.maxmoves = self.env.size[0] * self.env.size[1]
        self.reset()

