__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import randn, zeros
from random import choice, random, gauss
from evolution import Evolution

class GA(Evolution):
    """ Genetic algorithm """
    
    # selection schemes
    tournament = False
    tournamentsize = 2
    
    topproportion = 0.2
    
    elitism = False
    eliteproportion = 0.5
    elitesize = None # override with an exact number
    
    mutationprob = 0.1
    mutationStdDev = 0.5
    
    def initPopulation(self):
        self.currentpop = [self.x0]
        for dummy in range(self.popsize-1):
            self.currentpop.append(self.x0+randn(self.xdim))        
    
    def crossOver(self, parents, nbChildren):
        """ generate a number of children by doing 1-point cross-over """
        children = []
        for dummy in range(nbChildren):
            p1 = choice(parents)
            if self.xdim < 2:
                children.append(p1)
            else:
                p2 = choice(parents)
                point = choice(range(self.xdim-1))
                res = zeros(self.xdim)
                res[:point] = p1[:point]
                res[point:] = p2[point:]
                children.append(res)
        return children          
    
    def mutated(self, indiv):
        """ mutate some genes of the given individual """
        res = indiv.copy()
        for i in range(self.xdim):
            if random() < self.mutationprob:
                res[i] = indiv[i] + gauss(0, self.mutationStdDev)
        return res
                
    def selectionSize(self):
        """ the number of parents selected from the current population """
        return int(self.popsize * self.topproportion)
    
    def eliteSize(self):
        if self.elitism:
            if self.elitesize != None:
                return self.elitesize
            else:
                return int(self.popsize * self.eliteproportion)
        else:
            return 0
        
    def select(self):
        """ select some of the individuals of the population, taking into account their fitnesses 
        @return: list of selected parents """
        if not self.tournament:
            tmp = zip(self.fitnesses, self.currentpop)
            tmp.sort(key = lambda x: x[0])            
            tmp2 = list(reversed(tmp))[:self.selectionSize()]
            return map(lambda x: x[1], tmp2)
        else:
            # TODO: tournament selection
            raise NotImplementedError()
        
    def produceOffspring(self):
        """ produce offspring by selection, mutation and crossover. """
        parents = self.select()
        es = min(self.eliteSize(), self.selectionSize())
        self.currentpop = parents[:es]
        for child in self.crossOver(parents, self.popsize-es):
            self.currentpop.append(self.mutated(child))
        
        