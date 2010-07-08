__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import zeros

from twoplayergame import TwoPlayerGame

# TODO: factor out the similarities with the CaptureGame and Go.

class GomokuGame(TwoPlayerGame):
    """ The game of Go-Moku, alias Connect-Five. """

    BLACK = 1
    WHITE = -1
    EMPTY = 0

    startcolor = BLACK

    def __init__(self, size):
        """ the size of the board is a tuple, where each dimension must be minimum 5. """
        self.size = size
        assert size[0] >= 5
        assert size[1] >= 5
        self.reset()

    def _iterPos(self):
        """ an iterator over all the positions of the board. """
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                yield (i, j)

    def reset(self):
        """ empty the board. """
        TwoPlayerGame.reset(self)
        self.movesDone = 0
        self.b = {}
        for p in self._iterPos():
            self.b[p] = self.EMPTY

    def _fiveRow(self, color, pos):
        """ Is this placement the 5th in a row? """
        # TODO: more efficient...
        for dir in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            found = 1
            for d in [-1, 1]:
                for i in range(1, 5):
                    next = (pos[0] + dir[0] * i * d, pos[1] + dir[1] * i * d)
                    if (next[0] < 0 or next[0] >= self.size[0]
                        or next[1] < 0 or next[1] >= self.size[1]
                        or self.b[next] != color):
                        break
                    else:
                        found += 1
            if found >= 5:
                return True
        return False

    @property
    def indim(self):
        return self.size[0] * self.size[1]

    @property
    def outdim(self):
        return 2 * self.size[0] * self.size[1]

    def getBoardArray(self):
        """ an array with thow boolean values per position, indicating
        'white stone present' and 'black stone present' respectively. """
        a = zeros(self.outdim)
        for i, p in enumerate(self._iterPos()):
            if self.b[p] == self.WHITE:
                a[2 * i] = 1
            elif self.b[p] == self.BLACK:
                a[2 * i + 1] = 1
        return a

    def isLegal(self, c, pos):
        return self.b[pos] == self.EMPTY

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
            self._setStone(c, pos)
            if self.movesDone == self.size[0] * self.size[1]:
                # DRAW
                self.winner = self.DRAW
            return True

    def getSensors(self):
        """ just a list of the board position states. """
        return map(lambda x: x[1], sorted(self.b.items()))

    def __str__(self):
        s = ''
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                val = self.b[(i, j)]
                if val == self.EMPTY: s += ' _'
                elif val == self.BLACK: s += ' #'
                elif val == self.WHITE: s += ' *'
                else: s += ' ' + str(val)
            s += '\n'
        if self.winner:
            if self.winner == self.BLACK:
                w = 'Black (#)'
            elif self.winner == self.WHITE:
                w = 'White (*)'
            else:
                w = self.winner
            s += 'Winner: ' + w
            s += ' (moves done:' + str(self.movesDone) + ')\n'
        return s

    def _neighbors(self, pos):
        """ the 4 neighboring positions """
        res = []
        if pos[1] < self.size - 1: res.append((pos[0], pos[1] + 1))
        if pos[1] > 0: res.append((pos[0], pos[1] - 1))
        if pos[0] < self.size - 1: res.append((pos[0] + 1, pos[1]))
        if pos[0] > 0: res.append((pos[0] - 1, pos[1]))
        return res

    def _setStone(self, c, pos):
        """ set stone """
        self.b[pos] = c

    def getLegals(self, c):
        """ return all the legal positions for a color """
        return filter(lambda p: self.b[p] == self.EMPTY, self._iterPos())

    def getKilling(self, c):
        """ return all legal positions for a color that immediately kill the opponent. """
        return filter(lambda p: self._fiveRow(c, p), self.getLegals(c))

    def playToTheEnd(self, p1, p2):
        """ alternate playing moves between players until the game is over. """
        assert p1.color == -p2.color
        i = 0
        p1.game = self
        p2.game = self
        players = [p1, p2]
        while not self.gameOver():
            p = players[i]
            self.performAction(p.getAction())
            i = (i + 1) % 2


