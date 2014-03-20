__author__ = 'Tom Schaul, tom@idsia.ch'

from capturetask import CaptureGameTask
from pybrain.rl.environments.twoplayergames.capturegameplayers import ModuleDecidingPlayer
from pybrain.rl.environments.twoplayergames import CaptureGame
from pybrain.rl.environments.twoplayergames.capturegameplayers.captureplayer import CapturePlayer
from pybrain.structure.networks.custom.capturegame import CaptureGameNetwork


class RelativeCaptureTask(CaptureGameTask):
    """ returns the (anti-symmetric) relative score of p1 with respect to p2.
    (p1 and p2 are CaptureGameNetworks)
    The score depends on:
    - greedy play
    - play with fixed starting positions (only first stone)
    - moves-until-win or moves-until-defeat (winning faster is better)
    - play with noisy moves (e.g. adjusting softmax temperature)

    """

    # are networks provided?
    useNetworks = False

    # maximal number of games per evaluation
    maxGames = 3

    presetGamesProportion = 0.5

    minTemperature = 0
    maxTemperature = 0.2

    verbose = False

    # coefficient determining the importance of long vs. short games w.r. to winning/losing
    numMovesCoeff = 0.5

    def __init__(self, size, **args):
        self.setArgs(**args)
        self.size = size
        self.task = CaptureGameTask(self.size)
        self.env = self.task.env
        if self.presetGamesProportion > 0:
            self.sPos = self._fixedStartingPos()
            self.cases = int(len(self.sPos) / self.presetGamesProportion)
        else:
            self.cases = 1
        self.maxmoves = self.size * self.size
        self.minmoves = 3

    def __call__(self, p1, p2):
        self.temp = self.minTemperature
        if self.useNetworks:
            p1 = ModuleDecidingPlayer(p1, self.task.env, temperature=self.temp)
            p2 = ModuleDecidingPlayer(p2, self.task.env, temperature=self.temp)
        else:
            assert isinstance(p1, CapturePlayer)
            assert isinstance(p2, CapturePlayer)
            p1.game = self.task.env
            p2.game = self.task.env
        p1.color = CaptureGame.BLACK
        p2.color = -p1.color
        self.player = p1
        self.opponent = p2

        # the games with increasing temperatures and lower coefficients
        coeffSum = 0.
        score = 0.
        np = int(self.cases * (1 - self.presetGamesProportion))
        for i in range(self.maxGames):
            coeff = 1 / (10 * self.temp + 1)
            preset = None
            if self.cases > 1:
                if i % self.cases >= np:
                    preset = self.sPos[(i - np) % self.cases]
                elif i < self.cases:
                    # greedy, no need to repeat, just increase the coefficient
                    if i == 0:
                        coeff *= np
                    else:
                        continue
            res = self._oneGame(preset)
            score += coeff * res
            coeffSum += coeff
            if self.cases == 1 or (i % self.cases == 0 and i > 0):
                self._globalWarming()

        return score / coeffSum

    def _globalWarming(self):
        """ increase temperature """
        if self.temp == 0:
            self.temp = 0.02
        else:
            self.temp *= 1.5
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
            randPart = self.temp / (self.temp + 1)
            self.opponent.randomPartMoves = randPart
            self.player.randomPartMoves = randPart
            return True
        else:
            return False

    def _fixedStartingPos(self):
        """ a list of starting positions, not along the border, and respecting symmetry. """
        res = []
        if self.size < 3:
            return res
        for x in range(1, (self.size + 1) / 2):
            for y in range(x, (self.size + 1) / 2):
                res.append((x, y))
        return res

    def _oneGame(self, preset=None):
        """ a single black stone can be set as the first move. """
        self.env.reset()
        if preset != None:
            self.env._setStone(CaptureGame.BLACK, preset)
            self.env.movesDone += 1
            self.env.playToTheEnd(self.opponent, self.player)
        else:
            self.env.playToTheEnd(self.player, self.opponent)
        moves = self.env.movesDone
        win = self.env.winner == self.player.color
        if self.verbose:
            print('Preset:', preset, 'T:', self.temp, 'Win:', win, 'after', moves, 'moves.')
        res = 1 - self.numMovesCoeff * (moves - self.minmoves) / (self.maxmoves - self.minmoves)
        if win:
            return res
        else:
            return - res


if __name__ == '__main__':
    assert RelativeCaptureTask(5)._fixedStartingPos() == [(1, 1), (1, 2), (2, 2)]
    assert RelativeCaptureTask(8)._fixedStartingPos() == [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

    net1 = CaptureGameNetwork(hsize=1)
    net2 = CaptureGameNetwork(hsize=1)
    #print(net1.params)
    #print(net2.params)

    r = RelativeCaptureTask(5, maxGames=40, useNetworks=True,
                            presetGamesProportion=0.5)

    print(r(net1, net2))
    print(r(net2, net1))
    r.maxGames = 200
    print(r(net1, net2))
    print(r(net2, net1))



