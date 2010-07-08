__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.environments.episodic import EpisodicTask
from inspect import isclass
from pybrain.utilities import  Named
from pybrain.rl.environments.twoplayergames import CaptureGame
from pybrain.rl.environments.twoplayergames.capturegameplayers import RandomCapturePlayer, ModuleDecidingPlayer
from pybrain.rl.environments.twoplayergames.capturegameplayers.captureplayer import CapturePlayer
from pybrain.structure.modules.module import Module


class CaptureGameTask(EpisodicTask, Named):
    """ The task of winning the maximal number of capture games against a fixed opponent. """

    # first game, opponent is black
    opponentStart = True

    # on subsequent games, starting players are alternating
    alternateStarting = False

    # numerical reward value attributed to winning
    winnerReward = 1.

    # coefficient determining the importance of long vs. short games w.r. to winning/losing
    numMovesCoeff = 0.

    # average over some games for evaluations
    averageOverGames = 10

    noisy = True

    def __init__(self, size, opponent = None, **args):
        EpisodicTask.__init__(self, CaptureGame(size))
        self.setArgs(**args)
        if opponent == None:
            opponent = RandomCapturePlayer(self.env)
        elif isclass(opponent):
            # assume the agent can be initialized without arguments then.
            opponent = opponent(self.env)
        else:
            opponent.game = self.env
        if not self.opponentStart:
            opponent.color = CaptureGame.WHITE
        self.opponent = opponent
        self.maxmoves = self.env.size * self.env.size
        self.minmoves = 3
        self.reset()

    def reset(self):
        self.switched = False
        EpisodicTask.reset(self)
        if self.opponent.color == CaptureGame.BLACK:
            # first move by opponent
            EpisodicTask.performAction(self, self.opponent.getAction())

    def isFinished(self):
        res = self.env.gameOver()
        if res and self.alternateStarting and not self.switched:
            # alternate starting player
            self.opponent.color *= -1
            self.switched = True
        return res

    def getReward(self):
        """ Final positive reward for winner, negative for loser. """
        if self.isFinished():
            win = (self.env.winner != self.opponent.color)
            moves = self.env.movesDone
            res = self.winnerReward - self.numMovesCoeff * (moves -self.minmoves)/(self.maxmoves-self.minmoves)
            if not win:
                res *= -1
            if self.alternateStarting and self.switched:
                # opponent color has been inverted after the game!
                res *= -1
            return res
        else:
            return 0

    def performAction(self, action):
        EpisodicTask.performAction(self, action)
        if not self.isFinished():
            EpisodicTask.performAction(self, self.opponent.getAction())

    def f(self, x):
        """ If a module is given, wrap it into a ModuleDecidingAgent before evaluating it.
        Also, if applicable, average the result over multiple games. """
        if isinstance(x, Module):
            agent = ModuleDecidingPlayer(x, self.env, greedySelection = True)
        elif isinstance(x, CapturePlayer):
            agent = x
        else:
            raise NotImplementedError('Missing implementation for '+x.__class__.__name__+' evaluation')
        res = 0
        agent.game = self.env
        self.opponent.game = self.env
        for _ in range(self.averageOverGames):
            agent.color = -self.opponent.color
            x = EpisodicTask.f(self, agent)
            res += x
        return res / float(self.averageOverGames)



