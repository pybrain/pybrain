__author__ = 'Tom Schaul, tom@idsia.ch'


from scipy import zeros, argmax

from pybrain.rl.environments.twoplayergames import CaptureGame
from randomplayer import RandomCapturePlayer
from pybrain.utilities import drawIndex


class ModuleDecidingPlayer(RandomCapturePlayer):
    """ A Capture-game player that plays according to the rules, but choosing its moves
    according to the output of a module that takes as input the current state of the board. """

    greedySelection = False

    def __init__(self, module, *args, **kwargs):
        RandomCapturePlayer.__init__(self, *args, **kwargs)
        self.module = module
        
    def getAction(self):
        """ get suggested action, return them if they are legal, otherwise choose randomly. """ 
        ba = self.game.getBoardArray()
        # network is given inputs with self/other as input, not black/white
        if self.color != CaptureGame.BLACK:
            # invert values
            tmp = zeros(len(ba))
            tmp[:len(ba)-1:2] = ba[1:len(ba):2]
            tmp[1:len(ba):2] = ba[:len(ba)-1:2]
            ba = tmp
        return [self.color, self._legalizeIt(self.module.activate(ba))]
    
    def newEpisode(self):
        self.module.reset()

    def _legalizeIt(self, a):
        """ draw index from an array of probabilities, filtering out illegal moves. """
        assert min(a) >= 0
        legals = self.game.getLegals(self.color)
        probs = zeros(len(a))
        for i in map(self._convertPosToIndex, legals):
            probs[i] = a[i]
        # geedy selection
        if self.greedySelection:
            drawn = argmax(probs)
        else:
            drawn = drawIndex(probs, tolerant = True)
        return self._convertIndexToPos(drawn)
        
    def _convertIndexToPos(self, i):
        return (i/self.game.size, i%self.game.size)
    
    def _convertPosToIndex(self, p):
        return p[0]*self.game.size+p[1]
        
    def integrateObservation(self, obs = None):
        pass
 