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
    elitesize = 1
    
    mutationprob = 0.1
    mutationStdDev = 0.5
    
    def initPopulation(self):
        self.currentpop = []
        for dummy in range(self.popsize):
            self.currentpop.append(randn(self.xdim))        
    
    def crossOver(self, parents, nbChildren):
        """ generate a number of children by doing crosover """
        children = []
        for dummy in range(nbChildren):
            p1 = choice(parents)
            p2 = choice(parents)
            point = choice(range(self.xdim-1))
            res = zeros(self.xdim)
            res[:point] = p1[:point]
            res[point:] = p2[point:]
            children.append(res)
        return children          
    
    def mutate(self, indiv):
        """ mutate some genes of the given individual (in-place) """
        for i in range(self.xdim):
            if random() < self.mutationprob:
                indiv[i] += gauss(0, self.mutationStdDev)
                
    def selectionSize(self):
        """ the number of parents selected from the current population """
        return int(self.popsize * self.topproportion)
        
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
        # TODO: elitism
        self.currentpop = self.crossOver(parents, self.popsize)
        for child in self.currentpop:
            self.mutate(child)
        