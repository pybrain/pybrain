__author__ = 'Julian Togelius and Tom Schaul, tom@idsia.ch'

from random import shuffle

from pybrain.optimization.optimizer import BlackBoxOptimizer

    
class ES(BlackBoxOptimizer):
    """ Standard evolution strategy, (mu + lambda). """    
    
    mu = 50
    lambada = 50
    
    evaluatorIsNoisy = False
    
    storeHallOfFame = True
    
    mustMaximize = True
    
    def _additionalInit(self):        
        assert self.lambada % self.mu == 0, 'lambda ('+str(self.lambada)+\
                                            ') must be multiple of mu ('+str(self.mu)+').'
        self.hallOfFame = []        
        # population is a list of (fitness, individual) tuples.
        self.population = [(self._oneEvaluation(self._initEvaluable), self._initEvaluable)]
        for _ in range(1, self.mu + self.lambada):
            x = self._initEvaluable.copy()
            x.mutate()
            self.population.append((self._oneEvaluation(x), x))        
        self._sortPopulation()
                
    def _learnStep(self):               
        # re-evaluate the mu individuals if the fitness function is noisy        
        if self.evaluatorIsNoisy:
            for i in range (self.mu):
                x = self.population[i][1]
                self.population[i] = (self._oneEvaluation(x), x)
            self._sortPopulation(noHallOfFame = True)     
                   
        # generate the lambada: copy the mu and mu-tate the copies 
        for i in range(self.mu, self.mu + self.lambada):
            x = self.population[i % self.mu][1].copy()
            x.mutate()
            xFitness = self._oneEvaluation(x)
            self.population[i] = (xFitness, x)
        self._sortPopulation()      

    def _sortPopulation(self, noHallOfFame = False):
        # shuffle-sort the population and fitnesses
        shuffle(self.population)
        self.population.sort(key = lambda x: -x[0])
        if self.storeHallOfFame and not noHallOfFame:
            # the best per generation stored here
            self.hallOfFame.append(self.population[0][1])  
            
    @property
    def batchSize(self):
        if self.evaluatorIsNoisy:
            return self.mu + self.lambada
        else:
            return self.lambada
            
    def __str__(self):
        return 'ES('+str(self.mu)+'+'+str(self.lambada)+')'
