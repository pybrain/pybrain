__author__ = 'Tom Schaul, tom@idsia.ch'

from gomokutask import GomokuTask
from pybrain.rl.environments.twoplayergames.gomokuplayers import ModuleDecidingPlayer
from pybrain.rl.environments.twoplayergames import GomokuGame
from pybrain.rl.environments.twoplayergames.gomokuplayers.gomokuplayer import GomokuPlayer
from pybrain.structure.networks.custom.capturegame import CaptureGameNetwork


class RelativeGomokuTask(GomokuTask):
    """ returns the (anti-symmetric) relative score of p1 with respect to p2.
    (p1 and p2 are CaptureGameNetworks)
    The score depends on:
    - greedy play
    - moves-until-win or moves-until-defeat (winning faster is better)
    - play with noisy moves (e.g. adjusting softmax temperature)

    """

    # are networks provided?
    useNetworks = False

    # maximal number of games per evaluation
    maxGames = 3

    minTemperature = 0
    maxTemperature = 0.2

    verbose = False

    # coefficient determining the importance of long vs. short games w.r. to winning/losing
    numMovesCoeff = 0.5

    def __init__(self, size, **args):
        self.setArgs(**args)
        self.size = size
        self.task = GomokuTask(self.size)
        self.env = self.task.env
        self.maxmoves = self.env.size[0] * self.env.size[1]
        self.minmoves = 9


    def __call__(self, p1, p2):
        self.temp = self.minTemperature
        if self.useNetworks:
            p1 = ModuleDecidingPlayer(p1, self.task.env, temperature = self.temp)
            p2 = ModuleDecidingPlayer(p2, self.task.env, temperature = self.temp)
        else:
            assert isinstance(p1, GomokuPlayer)
            assert isinstance(p2, GomokuPlayer)
            p1.game = self.task.env
            p2.game = self.task.env
        p1.color = GomokuGame.BLACK
        p2.color = -p1.color
        self.player = p1
        self.opponent = p2

        # the games with increasing temperatures and lower coefficients
        coeffSum = 0.
        res = 0.
        for i in range(self.maxGames):
            coeff = 1/(10*self.temp+1)
            res += coeff * self._oneGame()
            coeffSum += coeff
            if i > 0:
                self._globalWarming()

        return res / coeffSum

    def _globalWarming(self):
        """ increase temperature """
        if self.temp == 0:
            self.temp = 0.02
        else:
            self.temp *= 1.2
        if self.temp > self.maxTemperature:
            return False
        elif self._setTemperature() == False:
            # not adjustable, keep it fixed then.
            self.temp = self.minTemperature
            return False
        return True

    def _setTemperature(self):
        if self.useNetworks:
            self.opponent.temperature = self.temp
            self.player.temperature = self.temp
            return True
        elif hasattr(self.opponent, 'randomPartMoves'):
            # an approximate conversion of temperature into random proportion:
            randPart = self.temp/(self.temp+1)
            self.opponent.randomPartMoves = randPart
            self.player.randomPartMoves = randPart
            return True
        else:
            return False

    def _oneGame(self, preset = None):
        """ a single black stone can be set as the first move. """
        self.env.reset()
        if preset != None:
            self.env._setStone(GomokuGame.BLACK, preset)
            self.env.movesDone += 1
            self.env.playToTheEnd(self.opponent, self.player)
        else:
            self.env.playToTheEnd(self.player, self.opponent)
        moves = self.env.movesDone
        win = self.env.winner == self.player.color
        if self.verbose:
            print('Preset:', preset, 'T:', self.temp, 'Win:', win, 'after', moves, 'moves.')
        res = 1 - self.numMovesCoeff * (moves -self.minmoves)/(self.maxmoves-self.minmoves)
        if win:
            return res
        else:
            return -res



if __name__ == '__main__':
    net1 = CaptureGameNetwork(hsize = 1)
    net2 = CaptureGameNetwork(hsize = 1)
    r = RelativeGomokuTask(7, maxGames = 10, useNetworks = True)
    print(r(net1, net2))
    print(r(net2, net1))
    print(r.env)
    r.maxGames = 50
    print(r(net1, net2))
    print(r(net2, net1))
    print(r.env)

