""" A basic test for competitive coevolution - single-parameter, linear fitness landscape,
transitive playing strength. """

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.learners.search.competitivecoevolution import CompetitiveCoevolution

from random import random

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
        
        
def Eva(x1, x2):
    return x1.strength > x2.strength

if __name__ == '__main__':
    x1 = Indiv()
    x2 = Indiv()
    x1.strength = -0.01
    x2.strength = 0.01
    L = CompetitiveCoevolution(Eva, x1, x2, populationSize = 10, verbose = True)
    print L.learn(1e4)
    print L.hallOfFame
