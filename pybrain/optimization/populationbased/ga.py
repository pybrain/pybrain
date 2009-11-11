__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import randn, zeros
from random import choice, random, gauss

from evolution import Evolution
from pybrain.optimization.optimizer import ContinuousOptimizer


class GA(ContinuousOptimizer, Evolution):
    """ Standard Genetic Algorithm. """
    
    #: selection scheme
    tournament = False
    tournamentSize = 2
    
    #: selection proportion
    topProportion = 0.2
    
    elitism = False
    eliteProportion = 0.5
    _eliteSize = None # override with an exact number
    
    #: mutation probability
    mutationProb = 0.1
    mutationStdDev = 0.5
    initRangeScaling = 10.
    
    initialPopulation = None
    
    mustMaximize = True
    
    def initPopulation(self):
        if self.initialPopulation is not None:
            self.currentpop = self.initialPopulation
        else:
            self.currentpop = [self._initEvaluable]
            for _ in range(self.populationSize-1):
                self.currentpop.append(self._initEvaluable+randn(self.numParameters)
                                       *self.mutationStdDev*self.initRangeScaling)        
    
    def crossOver(self, parents, nbChildren):
        """ generate a number of children by doing 1-point cross-over """
        xdim = self.numParameters
        children = []
        for _ in range(nbChildren):
            p1 = choice(parents)
            if xdim < 2:
                children.append(p1)
            else:
                p2 = choice(parents)
                point = choice(range(xdim-1))
                res = zeros(xdim)
                res[:point] = p1[:point]
                res[point:] = p2[point:]
                children.append(res)
        return children          
    
    def mutated(self, indiv):
        """ mutate some genes of the given individual """
        res = indiv.copy()
        for i in range(self.numParameters):
            if random() < self.mutationProb:
                res[i] = indiv[i] + gauss(0, self.mutationStdDev)
        return res
            
    @property    
    def selectionSize(self):
        """ the number of parents selected from the current population """
        return int(self.populationSize * self.topProportion)
    
    @property
    def eliteSize(self):
        if self.elitism:
            if self._eliteSize != None:
                return self._eliteSize
            else:
                return int(self.populationSize * self.eliteProportion)
        else:
            return 0
        
    def select(self):
        """ select some of the individuals of the population, taking into account their fitnesses
        
        :return: list of selected parents """
        if not self.tournament:
            tmp = zip(self.fitnesses, self.currentpop)
            tmp.sort(key = lambda x: x[0])            
            tmp2 = list(reversed(tmp))[:self.selectionSize]
            return map(lambda x: x[1], tmp2)
        else:
            # TODO: tournament selection
            raise NotImplementedError()
        
    def produceOffspring(self):
        """ produce offspring by selection, mutation and crossover. """
        parents = self.select()
        es = min(self.eliteSize, self.selectionSize)
        self.currentpop = parents[:es]
        for child in self.crossOver(parents, self.populationSize-es):
            self.currentpop.append(self.mutated(child))
            
        
        