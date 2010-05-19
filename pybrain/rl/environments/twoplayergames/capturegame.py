__author__ = 'Tom Schaul, tom@idsia.ch'

from random import choice
from scipy import zeros

from twoplayergame import TwoPlayerGame


# TODO: undo operation


class CaptureGame(TwoPlayerGame):
    """ the capture game is a simplified version of the Go game: the first player to capture a stone wins!
    Pass moves are forbidden."""
    # CHECKME: suicide allowed?

    BLACK = 1
    WHITE = -1
    EMPTY = 0

    startcolor = BLACK

    def __init__(self, size, suicideenabled=True):
        """ the size of the board is generally between 3 and 19. """
        self.size = size
        self.suicideenabled = suicideenabled
        self.reset()

    def _iterPos(self):
        """ an iterator over all the positions of the board. """
        for i in range(self.size):
            for j in range(self.size):
                yield (i, j)

    def reset(self):
        """ empty the board. """
        TwoPlayerGame.reset(self)
        self.movesDone = 0
        self.b = {}
        for p in self._iterPos():
            self.b[p] = self.EMPTY
        # which stone belongs to which group
        self.groups = {}
        # how many liberties does each group have
        self.liberties = {}

    @property
    def indim(self):
        return self.size ** 2

    @property
    def outdim(self):
        return 2 * self.size ** 2

    def getBoardArray(self):
        """ an array with two boolean values per position, indicating
        'white stone present' and 'black stone present' respectively. """
        a = zeros(self.outdim)
        for i, p in enumerate(self._iterPos()):
            if self.b[p] == self.WHITE:
                a[2 * i] = 1
            elif self.b[p] == self.BLACK:
                a[2 * i + 1] = 1
        return a

    def isLegal(self, c, pos):
        if pos not in self.b:
            return False
        elif self.b[pos] != self.EMPTY:
            return False
        elif not self.suicideenabled:
            return not self._suicide(c, pos)
        return True

    def doMove(self, c, pos):
        """ the action is a (color, position) tuple, for the next stone to move.
        returns True if the move was legal. """
        self.movesDone += 1
        if pos == 'resign':
            self.winner = -c
            return True
        elif not self.isLegal(c, pos):
            return False
        elif self._suicide(c, pos):
            assert self.suicideenabled
            self.b[pos] = 'y'
            self.winner = -c
            return True
        elif self._capture(c, pos):
            self.winner = c
            self.b[pos] = 'x'
            return True
        else:
            self._setStone(c, pos)
            return True

    def getSensors(self):
        """ just a list of the board position states. """
        return map(lambda x: x[1], sorted(self.b.items()))

    def __str__(self):
        s = ''
        for i in range(self.size):
            for j in range(self.size):
                val = self.b[(i, j)]
                if val == self.EMPTY: s += ' .'
                elif val == self.BLACK: s += ' X'
                elif val == self.WHITE: s += ' O'
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
        """ set stone, and update liberties and groups. """
        self.b[pos] = c
        merge = False
        self.groups[pos] = self.size * pos[0] + pos[1]
        freen = filter(lambda n: self.b[n] == self.EMPTY, self._neighbors(pos))
        self.liberties[self.groups[pos]] = set(freen)
        for n in self._neighbors(pos):
            if self.b[n] == -c:
                self.liberties[self.groups[n]].difference_update([pos])
            elif self.b[n] == c:
                if merge:
                    newg = self.groups[pos]
                    oldg = self.groups[n]
                    if newg == oldg:
                        self.liberties[newg].difference_update([pos])
                    else:
                        # merging 2 groups
                        for p in self.groups.keys():
                            if self.groups[p] == oldg:
                                self.groups[p] = newg
                        self.liberties[newg].update(self.liberties[oldg])
                        self.liberties[newg].difference_update([pos])
                        del self.liberties[oldg]
                else:
                    # connect to this group
                    del self.liberties[self.groups[pos]]
                    self.groups[pos] = self.groups[n]
                    self.liberties[self.groups[n]].update(freen)
                    self.liberties[self.groups[n]].difference_update([pos])
                    merge = True

    def _suicide(self, c, pos):
        """ would putting a stone here be suicide for c? """
        # any free neighbors?
        for n in self._neighbors(pos):
            if self.b[n] == self.EMPTY:
                return False

        # any friendly neighbor with extra liberties?
        for n in self._neighbors(pos):
            if self.b[n] == c:
                if len(self.liberties[self.groups[n]]) > 1:
                    return False

        # capture all surrounding ennemies?
        if self._capture(c, pos):
            return False

        return True

    def _capture(self, c, pos):
        """ would putting a stone here lead to a capture? """
        for n in self._neighbors(pos):
            if self.b[n] == -c:
                if len(self.liberties[self.groups[n]]) == 1:
                    return True
        return False

    def getLiberties(self, pos):
        """ how many liberties does the stone at pos have? """
        if self.b[pos] == self.EMPTY:
            return None
        return len(self.liberties[self.groups[pos]])

    def getGroupSize(self, pos):
        """ what size is the worm that this stone is part of? """
        if self.b[pos] == self.EMPTY:
            return None
        g = self.groups[pos]
        return len(filter(lambda x: x == g, self.groups.values()))

    def getLegals(self, c):
        """ return all the legal positions for a color """
        return filter(lambda p: self.b[p] == self.EMPTY, self._iterPos())

    def getAcceptable(self, c):
        """ return all legal positions for a color that don't commit suicide. """
        return filter(lambda p: not self._suicide(c, p), self.getLegals(c))

    def getKilling(self, c):
        """ return all legal positions for a color that immediately kill the opponent. """
        return filter(lambda p: self._capture(c, p), self.getAcceptable(c))

    def randomBoard(self, nbmoves):
        """ produce a random, undecided and legal capture-game board, after at most nbmoves.
        :return: the number of moves actually done. """
        c = self.BLACK
        self.reset()
        for i in range(nbmoves):
            l = set(self.getAcceptable(c))
            l.difference_update(self.getKilling(c))
            if len(l) == 0:
                return i
            self._setStone(c, choice(list(l)))
            c = -c
        return nbmoves

    def giveHandicap(self, h, color=BLACK):
        i = 0
        for pos in self._handicapIterator():
            i += 1
            if i > h:
                return
            if self.isLegal(color, pos):
                self._setStone(color, pos)

    def _handicapIterator(self):
        s = self.size
        assert s > 2
        yield (1, 1)
        if s > 3:
            # 4 corners
            yield (s - 2, s - 2)
            yield (1, s - 2)
            yield (s - 2, 1)
        if s > 4:
            for i in range(2, s - 2):
                yield (i, 1)
                yield (i, s - 2)
                yield (1, i)
                yield (s - 2, i)

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

