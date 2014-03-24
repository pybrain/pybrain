__author__ = 'Tom Schaul, tom@idsia.ch'

from random import choice

from pybrain.optimization.coevolution.coevolution import Coevolution


class MultiPopulationCoevolution(Coevolution):
    """ Coevolution with a number of independent populations. """

    numPops = 10

    def __str__(self):
        return 'MultiPop'+str(self.numPops)+Coevolution.__str__(self)

    def _initPopulation(self, seeds):
        """ one part of the seeds for each population, if there's not enough: randomize. """
        for s in seeds:
            s.parent = None
        while len(seeds) < self.numPops:
            tmp = choice(seeds).copy()
            tmp.randomize()
            seeds.append(tmp)
        self.pops = []
        for i in range(self.numPops):
            si = seeds[i::self.numPops]
            self.pops.append(self._extendPopulation(si, self.populationSize))
        self.mainpop = 0
        self.pop = self.pops[self.mainpop]

    def _evaluatePopulation(self):
        """Each individual in main pop plays against
        tournSize others of each other population (the best part of them). """
        for other in self.pops:
            if other == self.pop:
                continue
            # TODO: parametrize
            bestPart = len(other)/2
            if bestPart < 1:
                bestPart = 1
            self._doTournament(self.pop, other[:bestPart], self.tournamentSize)

        fitnesses = []
        for p in self.pop:
            fit = 0
            for other in self.pops:
                if other == self.pop:
                    continue
                for opp in other:
                    fit += self._beats(p, opp)
            if self.absEvalProportion > 0 and self.absEvaluator != None:
                fit = (1-self.absEvalProportion) * fit + self.absEvalProportion * self.absEvaluator(p)
            fitnesses.append(fit)
        return fitnesses

    def _oneGeneration(self):
        Coevolution._oneGeneration(self)
        # change the main pop
        self.pops[self.mainpop] = self.pop
        self.mainpop = self.generation % self.numPops
        self.pop = self.pops[self.mainpop]

    def _stepsPerGeneration(self):
        if self.tournamentSize == None:
            return 2 * self.populationSize** 2
        else:
            return Coevolution._stepsPerGeneration(self)
