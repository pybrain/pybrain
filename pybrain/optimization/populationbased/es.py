__author__ = 'Julian Togelius and Tom Schaul, tom@idsia.ch'

from random import shuffle

from pybrain.optimization.optimizer import BlackBoxOptimizer


class ES(BlackBoxOptimizer):
    """ Standard evolution strategy, (mu +/, lambda). """

    mu = 50
    lambada = 50

    evaluatorIsNoisy = False

    storeHallOfFame = True

    mustMaximize = True

    elitism = False

    def _additionalInit(self):
        assert self.lambada % self.mu == 0, 'lambda ('+str(self.lambada)+\
                                            ') must be multiple of mu ('+str(self.mu)+').'
        self.hallOfFame = []
        # population is a list of (fitness, individual) tuples.
        self.population = [(self._oneEvaluation(self._initEvaluable), self._initEvaluable)] * self._popsize
        map(self._replaceByMutation, range(1, self._popsize))
        self._sortPopulation()

    @property
    def _popsize(self):
        if self.elitism:
            return self.mu + self.lambada
        else:
            return self.lambada

    def _replaceByMutation(self, index):
        x = self.population[index][1].copy()
        x.mutate()
        self.population[index] = (self._oneEvaluation(x), x)

    def _learnStep(self):
        # re-evaluate the mu individuals if the fitness function is noisy
        if self.evaluatorIsNoisy:
            for i in range (self.mu):
                x = self.population[i][1]
                self.population[i] = (self._oneEvaluation(x), x)
            self._sortPopulation(noHallOfFame = True)

        # produce offspring from the the mu best ones
        self.population = self.population[:self.mu] * (self._popsize/self.mu)

        # mutate the offspring
        if self.elitism:
            map(self._replaceByMutation, range(self.mu, self._popsize))
        else:
            map(self._replaceByMutation, range(self._popsize))

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
            return self._popsize
        else:
            return self.lambada

    def __str__(self):
        if self.elitism:
            return 'ES('+str(self.mu)+'+'+str(self.lambada)+')'
        else:
            return 'ES('+str(self.mu)+','+str(self.lambada)+')'
