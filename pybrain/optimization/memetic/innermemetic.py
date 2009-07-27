__author__ = 'Tom Schaul, tom@idsia.ch'

from memetic import MemeticSearch
from pybrain.rl.learners.search.es import ES


class InnerMemeticSearch(MemeticSearch, ES):
    """ Population-based memetic search """
    
    mu = 5
    lambada = 5
    
    def __init__(self, *args, **kwargs):
        MemeticSearch.__init__(self, *args, **kwargs)
        
        # lambada must be mu-ltiple of mu
        assert self.lambada % self.mu == 0
        
        # population is a list of (fitness, individual) tuples.
        self.population = [(self.bestEvaluation, self.bestEvaluable.copy())]
        for dummy in range(1, self.mu + self.lambada):
            x = self.bestEvaluable.copy()
            x.mutate()
            self.population.append((self.evaluator(x), x))
        
        self._sortPopulation()
        self.steps = self.mu + self.lambada
        
    def _learnStep(self):
        # do a step only if we have accumulated the resources to do a whole batch.
        if self.noisy:
            if self.steps % (self.mu + self.lambada*self.localSteps) != 0:
                return
        else:
            if self.steps % (self.lambada*self.localSteps) != 0:
                return
        
        # re-evaluate the mu individuals if the fitness function is noisy        
        if self.noisy:
            for i in range (self.mu):
                x = self.population[i][1]
                self.population[i] = (self.evaluator(x), x)
            self._sortPopulation()
            
        # generate the lambada: copy the mu and mu-tate the copies 
        for i in range(self.mu, self.mu + self.lambada):
            x = self.population[i % self.mu][1].copy()
            x.topologyMutate()
            outsourced = self.localSearch(self.evaluator, x, maxEvaluations = self.localSteps, 
                                          desiredEvaluation = self.desiredEvaluation,
                                          **self.localSearchArgs)
            x, xFitness = outsourced.learn()
            self.population[i] = (xFitness, x)
            if self.desiredEvaluation != None and xFitness >= self.desiredEvaluation:
                self.bestEvaluable, self.bestEvaluation = x, xFitness
                return

        self._sortPopulation()
        if self.verbose:
            print self.steps, ':', self.population[0][0]
        