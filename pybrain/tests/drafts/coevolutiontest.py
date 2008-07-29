""" A basic test for competitive coevolution - single-parameter, linear fitness landscape,
transitive playing strength. """

__author__ = 'Tom Schaul, tom@idsia.ch'

from random import random

from pybrain.rl.learners.search import CompetitiveCoevolution, Coevolution

class Indiv:
    strength = 0.
    
    def mutate(self):
        self.strength += (random()-0.5)*0.02
        
    def randomize(self):
        self.strength = (random()-0.5)*2
    
    def copy(self):
        res = Indiv()
        res.strength = self.strength
        return res
    
    def __repr__(self):
        return 'I'+str(int(1e3*self.strength))
    
    @property
    def params(self):
        return self.strength
    
    def _setParameters(self, x):
        self.strength = x
        
class RockPaperScissors:
    rockval = 0
    paperval = 0
    scissorval = 0
    
    def mutate(self):
        self.rockval += int((random()-0.5)*10)
        self.paperval += int((random()-0.5)*10)
        self.scissorval += int((random()-0.5)*10)
        
    def randomize(self):
        self.rockval = int((random()-0.5)*20)
        self.paperval = int((random()-0.5)*20)
        self.scissorval = int((random()-0.5)*20)
        
    def copy(self):
        res = RockPaperScissors()
        res.rockval = self.rockval
        res.paperval = self.paperval
        res.scissorval = self.scissorval
        return res
    
    def __repr__(self):
        return '{'+str(self.rockval)+'|'+str(self.paperval)+'|'+str(self.scissorval)+'}'
    
    @property
    def params(self):
        return [self.rockval, self.paperval, self.scissorval]
    
    def _setParameters(self, x):
        self.rockval, self.paperval, self.scissorval = x
        
    def _isAmbiguous(self):
        return self.rockval == self.paperval or self.rockval == self.scissorval or self.paperval == self.scissorval
            
    def _isRock(self):
        return self.rockval >= self.paperval and self.rockval >= self.scissorval
    
    def _isPaper(self):
        return self.paperval >= self.rockval and self.paperval >= self.scissorval
    
    def _isScissor(self):
        return self.scissorval >= self.rockval and self.scissorval >= self.paperval
        
    
    
def RPSEval(x1, x2):
    if x1._isAmbiguous():
        if x2._isAmbiguous():
            win = sum(x1.params) > sum(x2.params)
        else:
            win = True
    elif x2._isAmbiguous():
        win = False
    else:
        if x1._isRock() and x2._isPaper():
            win = False
        if x2._isRock() and x1._isPaper():
            win = True
        if x1._isScissor() and x2._isPaper():
            win = True
        if x2._isScissor() and x1._isPaper():
            win = False
        if x1._isRock() and x2._isScissor():
            win = True
        if x2._isRock() and x1._isScissor():
            win = False
        if x1._isRock() and x2._isRock():
            win = x1.rockval > x2.rockval
        if x1._isPaper() and x2._isPaper():
            win = x1.paperval > x2.paperval
        if x1._isScissor() and x2._isScissor():
            win = x1.scissorval > x2.scissorval            
    if win: 
        return 1
    else:
        return -1
    
        
def Eva(x1, x2):
    if random() < 0.1:
        return Eva(x2, x1)
    elif x1.strength > x2.strength:
        return 1
    else:
        return -1


if False:
    x1 = Indiv()
    x2 = Indiv()
    x1.strength = -0.01
    x2.strength = 0.01
    
    L = CompetitiveCoevolution(Eva, [x1], 
    #L = Coevolution(Eva, [x1, x2], 
                    populationSize = 10, 
                    verbose = True,
                    #selectionProportion = 0.2,
                    #parentChildAverage = 0.5,
                    tournamentSize = 2,
                    elitism = False,
                    )
    print L
    print L.learn(1e3)
    print L.hallOfFame
    
    
if __name__ == '__main__':
    x1 = RockPaperScissors()
    x1.randomize()
    print x1
    L = CompetitiveCoevolution(RPSEval, [x1], 
                               populationSize = 4, 
                               selectionProportion = 0.25,
                               verbose = True,
                               tournamentSize = 4,
                               elitism = False,
                               useSharedSampling = True,
                               hallOfFameEvaluation = 0.75
                               )
    print L
    print L.learn(2e4)
    
    import pylab
    from resultreader import slidingAverage
    numPops = 2
    avgOver = 5
    avgRelFits = map(lambda x: sum(x)/len(x), L.hallOfFitnesses)
    bestRelFits = map(max, L.hallOfFitnesses)
    hm = ['-', '.-', 'o', '.', ':']
    pylab.title('Relative averaged '+str(avgOver))
    for g in range(numPops):
        pylab.plot(slidingAverage(avgRelFits[g::numPops], avgOver), hm[g%numPops], label = 'avg'+str(g+1))
        pylab.plot(slidingAverage(bestRelFits[g::numPops], avgOver), hm[g%numPops], label = 'max'+str(g+1))
    pylab.legend()
    pylab.show()
    
    
    