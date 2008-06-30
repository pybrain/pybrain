__author__ = 'Julian Togelius and Tom Schaul, tom@idsia.ch'

from random import shuffle

from pybrain.rl.learners.learner import Learner

    
class ES(Learner):
    """ Standard evolution strategy, (mu + lambda). """    
    
    mu = 50
    lambada = 50
    
    noisy = False
     
    def __init__(self, evaluator, evaluable, **args):
        Learner.__init__(self, evaluator, evaluable, **args)
        
        # lambada must be mu-ltiple of mu
        assert self.lambada % self.mu == 0
        
        # population is a list of (fitness, individual) tuples.
        self.population = [(self.bestEvaluation, self.bestEvaluable.copy())]
        for dummy in range(1, self.mu + self.lambada):
            x = self.bestEvaluable.copy()
            x.mutate()
            self.population.append((self.evaluator(x), x))
        
        self._sortPopulation()
        self.steps = self.mu+self.lambada
        
    def _learnStep(self):       
        # do a step only if we have accumulated the resources to do a whole batch.
        if self.steps % self._stepsPerGeneration() != 0: return
        
        # re-evaluate the mu individuals if the fitness function is noisy        
        if self.noisy:
            for i in range (self.mu):
                x = self.population[i][1]
                self.population[i] = (self.evaluator(x), x)
            self._sortPopulation()
            
        # generate the lambada: copy the mu and mu-tate the copies 
        for i in range(self.mu, self.mu + self.lambada):
            x = self.population[i % self.mu][1].copy()
            x.mutate()
            xFitness = self.evaluator(x)
            self.population[i] = (xFitness, x)
            if self.desiredEvaluation != None and xFitness >= self.desiredEvaluation:
                self.bestEvaluable, self.bestEvaluation = x, xFitness
                return

        self._sortPopulation()
        if self.verbose:
            print self.steps, ':', self.population[0][0]

    def _sortPopulation(self):
        # shuffle-sort the population and fitnesses
        shuffle(self.population)
        self.population.sort(key = lambda x: -x[0])
        
        if self.population[0][0] >= self.bestEvaluation:
            self.bestEvaluation, self.bestEvaluable = self.population[0]
            
    def _stepsPerGeneration(self):            
        if self.noisy:
            return self.mu + self.lambada
        else:
            return self.lambada                
        
        