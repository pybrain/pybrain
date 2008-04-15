__author__ = 'Julian Togelius and Tom Schaul, tom@idsia.ch'

from random import shuffle

from searchprocess import SearchProcess

    
class ES(SearchProcess):
    """ Standard evolution strategy, mu + lambda. """    
    
    mu = 50
    lambada = 50
    noisy = True
     
    def __init__(self, evolvable, evaluator, desiredFitness = None, mu = 5, lambada = 5):
        self.mu = mu
        self.lambada = lambada
        self.desiredFitness = desiredFitness
        self.steps = 0
        self.population = []
        self.evaluator = evaluator
        self.bestFitness = self.evaluator(evolvable) 
        self.population.append ((self.bestFitness, evolvable.copy()))
        for i in range(1, self.mu + self.lambada):
            x = evolvable.copy ()
            x.mutate ()
            self.population.append((self.evaluator(x), x))
        self.bestEvolvable = self.population[0][1].copy ()
     
    def _oneStep(self, verbose = False):       
        # evaluate the mu individuals if the fitness function is noisy        
        if self.noisy:
            for i in range (self.mu):
                x = self.population[i][1]
                self.population[i] = (self.evaluator(x), x)

        # shuffle-sort the population and fitnesses
        shuffle(self.population)
        self.population.sort(key = lambda x: -x[0])
        
        # generate the lambada: copy the mu and mu-tate the copies 
        for i in range(self.mu, self.mu + self.lambada):
            x = self.population[i % self.mu][1].copy()
            x.mutate()
            self.population[i] = (self.evaluator(x), x)
        # set convenience variables
        self.bestFitness = self.population[0][0]
        self.bestEvolvable = self.population[0][1].copy()
        if verbose:
            print self.steps, ':', self.bestFitness
            
            