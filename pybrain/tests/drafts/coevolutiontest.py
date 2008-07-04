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
        
        
        
def Eva(x1, x2):
    if random() < 0.1:
        return Eva(x2, x1)
    elif x1.strength > x2.strength:
        return 1
    else:
        return -1


if __name__ == '__main__':
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