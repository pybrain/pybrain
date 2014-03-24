__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import randn, zeros
from scipy import random as rd, array
from random import choice, random, gauss, shuffle, sample
from numpy import ndarray

from pybrain.optimization.populationbased.evolution import Evolution
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

    '''added by JPQ'''
    def initBoundaries(self):
        assert len(self.xBound) == self.numParameters
        self.mins = array([min_ for min_, max_ in self.xBound])
        self.maxs = array([max_ for min_, max_ in self.xBound])
    # ---        
    def initPopulation(self):
        '''added by JPQ '''
        if self.xBound is not None:
            self.initBoundaries()
        # ---
        if self.initialPopulation is not None:
            self.currentpop = self.initialPopulation
        else:
            self.currentpop = [self._initEvaluable]
            for _ in range(self.populationSize-1):
                '''added by JPQ '''
                if self.xBound is None:
                # ---
                    self.currentpop.append(self._initEvaluable+randn(self.numParameters)
                                       *self.mutationStdDev*self.initRangeScaling)
                    '''added by JPQ '''
                else:
                    position = rd.random(self.numParameters)
                    position *= (self.maxs-self.mins)
                    position += self.mins
                    self.currentpop.append(position)
                    # ---

    def crossOverOld(self, parents, nbChildren):
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
                point += 1
                res = zeros(xdim)
                res[:point] = p1[:point]
                res[point:] = p2[point:]
                children.append(res)
        return children
        
    def mutatedOld(self, indiv):
        """ mutate some genes of the given individual """
        res = indiv.copy()
        for i in range(self.numParameters):
            if random() < self.mutationProb:
                res[i] = indiv[i] + gauss(0, self.mutationStdDev)
        return res
        
    ''' added by JPQ in replacement of crossover and mutated '''    
    def crossOver(self, parents, nbChildren):
        """ generate a number of children by doing 1-point cross-over """
        """ change as the <choice> return quite often the same p1 and even
            several time p2 was return the same than p1 """
        xdim = self.numParameters
        shuffle(parents)
        children = []
        for i in range(len(parents)/2):
            p1 = parents[i]
            p2 = parents[i+(len(parents)/2)]
            if xdim < 2:
                children.append(p1)
                children.append(p2)
            else:
                point = choice(range(xdim-1))
                point += 1
                res = zeros(xdim)
                res[:point] = p1[:point]
                res[point:] = p2[point:]
                children.append(res)
                res = zeros(xdim)
                res[:point] = p2[:point]
                res[point:] = p1[point:]
                children.append(res)
        shuffle(children)
        if len(children) > nbChildren:
            children = children[:nbChildren]  
        elif len(children) < nbChildren:
            children +=sample(children,(nbChildren-len(children)))  
        return children
        
    def childexist(self,indiv,pop):
        if isinstance(pop,list):
            for i in range(len(pop)):
                if all((abs(indiv[k] - pop[i][k])/(self.maxs[k]-self.mins[k]))
                        < 1.e-7 for k in xrange(self.numParameters)):
                    return True
        return False
        
    def mutated(self, indiv):
        """ mutate some genes of the given individual """
        res = indiv.copy()
        #to avoid having a child identical to one of the currentpopulation'''
        for i in range(self.numParameters):
            if random() < self.mutationProb:
                if self.xBound is None:
                    res[i] = indiv[i] + gauss(0, self.mutationStdDev)
                else:
                    res[i] = max(min(indiv[i] + gauss(0, self.mutationStdDev),self.maxs[i]),
                             self.mins[i])
        return res

    def old_jpq_mutated(self, indiv, pop):
        """ mutate some genes of the given individual """
        res = indiv.copy()
        #to avoid having a child identical to one of the currentpopulation'''
        in_pop = self.childexist(indiv,pop)
        for i in range(self.numParameters):
            if random() < self.mutationProb:
                res[i] = max(min(indiv[i] + gauss(0, self.mutationStdDev),self.maxs[i]),
                             self.mins[i])
            
            if random() < self.mutationProb or in_pop:
                if self.xBound is None:
                    res[i] = indiv[i] + gauss(0, self.mutationStdDev)
                else:
                    if in_pop:
                        cmin = abs(indiv[i] - self.mins[i])/(self.maxs[i]-self.mins[i])
                        cmax = abs(indiv[i] - self.maxs[i])/(self.maxs[i]-self.mins[i])
                        if cmin < 1.e-7 or cmax < 1.e-7:
                            res[i] = self.mins[i] + random()*random()*(self.maxs[i]-self.mins[i])
                        else:
                            res[i] = max(min(indiv[i] + gauss(0, self.mutationStdDev),self.maxs[i]),
                             self.mins[i])
                    else:
                        res[i] = max(min(indiv[i] + gauss(0, self.mutationStdDev),self.maxs[i]),
                             self.mins[i])

        return res
    # ---
    
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
        '''Modified by JPQ '''
        nbchildren = self.populationSize - es
        if self.populationSize - es <= 0:
            nbchildren = len(parents)
        for child in self.crossOver(parents, nbchildren ):
            self.currentpop.append(self.mutated(child))
        # ---

