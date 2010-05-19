__author__ = 'Tom Schaul, tom@idsia.ch'


from inspect import isclass
from pybrain.utilities import  Named
from pybrain.rl.environments.twoplayergames import GomokuGame
from pybrain.rl.environments.twoplayergames.gomokuplayers import RandomGomokuPlayer, ModuleDecidingPlayer
from pybrain.rl.environments.twoplayergames.gomokuplayers.gomokuplayer import GomokuPlayer
from pybrain.structure.modules.module import Module
from pybrain.rl.environments.episodic import EpisodicTask


class GomokuTask(EpisodicTask, Named):
    """ The task of winning the maximal number of Gomoku games against a fixed opponent. """

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
        EpisodicTask.__init__(self, GomokuGame((size, size)))
        self.setArgs(**args)
        if opponent == None:
            opponent = RandomGomokuPlayer(self.env)
        elif isclass(opponent):
            # assume the agent can be initialized without arguments then.
            opponent = opponent(self.env)
        if not self.opponentStart:
            opponent.color = GomokuGame.WHITE
        self.opponent = opponent
        self.minmoves = 9
        self.maxmoves = self.env.size[0] * self.env.size[1]
        self.reset()

    def reset(self):
        self.switched = False
        EpisodicTask.reset(self)
        if self.opponent.color == GomokuGame.BLACK:
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
            if self.env.winner == self.env.DRAW:
                return 0
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
        elif isinstance(x, GomokuPlayer):
            agent = x
        else:
            raise NotImplementedError('Missing implementation for '+x.__class__.__name__+' evaluation')
        res = 0
        agent.game = self.env
        self.opponent.game = self.env
        for dummy in range(self.averageOverGames):
            agent.color = -self.opponent.color
            res += EpisodicTask.f(self, agent)
        return res / float(self.averageOverGames)



