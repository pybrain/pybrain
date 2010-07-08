__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.environments.twoplayergames.gomoku import GomokuGame


class PenteGame(GomokuGame):
    """ The game of Pente.

    The rules are similar to Go-Moku, except that it is now possible to capture
    stones, in pairs, by putting stones at both ends of a pair of the opponent.
    The game is won by the first player who either has 5 connected stones, or
    has captured 5 pairs.
    """

    def reset(self):
        GomokuGame.reset(self)
        self.pairsTaken = {self.BLACK: 0, self.WHITE: 0}
        center = (self.size[0] / 2, self.size[1] / 2)
        self._setStone(-self.startcolor, center)
        self.movesDone += 1

    def getKilling(self, c):
        """ return all legal positions for a color that immediately kill the opponent. """
        res = GomokuGame.getKilling(self, c)
        for p in self.getLegals(c):
            k = self._killsWhich(c, p)
            if self.pairsTaken[c] + len(k) / 2 >= 5:
                res.append(p)
        return res

    def _killsWhich(self, c, pos):
        """ placing a stone of color c at pos would kill which enemy stones? """
        res = []
        for dir in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            for d in [-1, 1]:
                killcands = []
                for i in [1, 2, 3]:
                    next = (pos[0] + dir[0] * i * d, pos[1] + dir[1] * i * d)
                    if (next[0] < 0 or next[0] >= self.size[0]
                        or next[1] < 0 or next[1] >= self.size[1]):
                        break
                    if i == 3 and self.b[next] == c:
                        res += killcands
                        break
                    if i != 3 and self.b[next] != -c:
                        break
                    killcands.append(next)
        return res

    def doMove(self, c, pos):
        """ the action is a (color, position) tuple, for the next stone to move.
        returns True if the move was legal. """
        self.movesDone += 1
        if not self.isLegal(c, pos):
            return False
        elif self._fiveRow(c, pos):
            self.winner = c
            self.b[pos] = 'x'
            return True
        else:
            tokill = self._killsWhich(c, pos)
            if self.pairsTaken[c] + len(tokill) / 2 >= 5:
                self.winner = c
                self.b[pos] = 'x'
                return True

            self._setStone(c, pos, tokill)
            if self.movesDone == (self.size[0] * self.size[1]
                                  + 2 * (self.pairsTaken[self.BLACK] + self.pairsTaken[self.WHITE])):
                # DRAW
                self.winner = self.DRAW
            return True

    def _setStone(self, c, pos, tokill=None):
        """ set stone, and potentially kill stones. """
        if tokill == None:
            tokill = self._killsWhich(c, pos)
        GomokuGame._setStone(self, c, pos)
        for p in tokill:
            self.b[p] = self.EMPTY
        self.pairsTaken[c] += len(tokill) / 2

    def __str__(self):
        s = GomokuGame.__str__(self)
        s += 'Black captured:' + str(self.pairsTaken[self.BLACK]) + ', white captured:' + str(self.pairsTaken[self.WHITE]) + '.'
        return s

