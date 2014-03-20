__author__ = 'Tom Schaul, tom@idsia.ch'


from scipy import zeros, ones

from pybrain.rl.environments.twoplayergames import GomokuGame
from randomplayer import RandomGomokuPlayer
from pybrain.utilities import drawGibbs


class ModuleDecidingPlayer(RandomGomokuPlayer):
    """ A Go-Moku player that plays according to the rules, but choosing its moves
    according to the output of a module that takes as input the current state of the board. """

    greedySelection = False

    # if the selection is not greedy, use Gibbs-sampling with this temperature
    temperature = 1.

    def __init__(self, module, *args, **kwargs):
        RandomGomokuPlayer.__init__(self, *args, **kwargs)
        self.module = module
        if self.greedySelection:
            self.temperature = 0.

    def getAction(self):
        """ get suggested action, return them if they are legal, otherwise choose randomly. """
        ba = self.game.getBoardArray()
        # network is given inputs with self/other as input, not black/white
        if self.color != GomokuGame.BLACK:
            # invert values
            tmp = zeros(len(ba))
            tmp[:len(ba)-1:2] = ba[1:len(ba):2]
            tmp[1:len(ba):2] = ba[:len(ba)-1:2]
            ba = tmp
        self.module.reset()
        return [self.color, self._legalizeIt(self.module.activate(ba))]

    def newEpisode(self):
        self.module.reset()

    def _legalizeIt(self, a):
        """ draw index from an array of values, filtering out illegal moves. """
        if not min(a) >= 0:
            print(a)
            print(min(a))
            print(self.module.params)
            print(self.module.inputbuffer)
            print(self.module.outputbuffer)
            raise Exception('No positve value in array?')
        legals = self.game.getLegals(self.color)
        vals = ones(len(a))*(-100)*(1+self.temperature)
        for i in map(self._convertPosToIndex, legals):
            vals[i] = a[i]
        drawn = self._convertIndexToPos(drawGibbs(vals, self.temperature))
        assert drawn in legals
        return drawn

    def _convertIndexToPos(self, i):
        return (i/self.game.size[0], i%self.game.size[0])

    def _convertPosToIndex(self, p):
        return p[0]*self.game.size[0]+p[1]

    def integrateObservation(self, obs = None):
        pass


