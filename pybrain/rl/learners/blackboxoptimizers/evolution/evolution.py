__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import argmax, array

from pybrain.rl.learners.blackboxoptimizers import BlackBoxOptimizer
from pybrain.utilities import abstractMethod


class Evolution(BlackBoxOptimizer):
    """ Base class for evolutionary algorithms, seen as function optimizers. """
    
    maxgenerations = 1e6
    
    popsize = 10
    
    # evolution generally tries to maximize fitness, not minimize a function
    minimize = False
    
    def __init__(self, f, **args):
        BlackBoxOptimizer.__init__(self, f, **args)
        assert self.minimize == False
        
        # current population
        self.currentpop = []
        self.fitnesses = []
        
        # for analysis purposes, store all kinds of stuff
        self.allgenerations = []
        
        self.bestfitness = -1e100
        self.bestx = None
        
    def stoppingCriterion(self):
        return self.bestfitness >= self.stopPrecision
        
    def optimize(self):    
        """ the main loop """
        self.initPopulation()
        self.generation = 0
        while not self.stoppingCriterion():
            if self.generation > self.maxgenerations:
                break
            if len(self.tfun.vallist) > self.maxEvals:
                break
            self.oneGeneration()
            self.generation += 1
            print 'Gen:', self.generation, 'fit:', self.bestfitness
        return self.bestx    
    
    def initPopulation(self):
        """ initialize the population """
        abstractMethod()
    
    def oneGeneration(self):
        """ do one generation step """
        # evaluate fitness
        self.fitnesses = []
        for indiv in self.currentpop:
            self.fitnesses.append(self.targetfun(indiv))
        
        # determine the best values
        best = argmax(array(self.fitnesses))
        self.bestfitness = self.fitnesses[best]
        self.bestx = self.currentpop[best]
        
        self.allgenerations.append((self.currentpop, self.fitnesses))
        
        self.produceOffspring()
        
    def produceOffspring(self):
        """ generate the new generation of offspring, given the current population, and their fitnesses """        
        abstractMethod()
    