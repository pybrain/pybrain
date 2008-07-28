__author__ = 'Tom Schaul, tom@idsia.ch'

from coevolution import Coevolution


class CompetitiveCoevolution(Coevolution):
    """ Coevolution with 2 independent populations, and competitive fitness sharing. """
    
    def __str__(self):
        return 'Competitive'+Coevolution.__str__(self)
    
    def _initPopulation(self, seeds):
        """ one half for each population """
        for s in seeds:
            s.parent = None
        if len(seeds) > 1:
            s1 = seeds[:len(seeds)/2]
            s2 = seeds[len(seeds)/2:]
        else:
            # not enough seeds: randomize 
            s1 = seeds
            s2 = [seeds[0].copy().randomize()]
        self.pop = self._extendPopulation(s1, self.populationSize)
        self.parasitePop = self._extendPopulation(s2, self.populationSize)
        
    def _competitiveSharedFitness(self, hosts, parasites):
        """ determine the competitive shared fitness for the population of hosts, w.r. to
        the population of parasites. """
        # determine beat-sum for parasites (nb of games lost)
        beatsums = {}
        for p in parasites:
            beatsums[p] = 0
            for h in hosts:
                beatsums[p] += self._beats(h, p)
                
        # determine fitnesses for hosts
        fitnesses = []
        for h in hosts:
            hsum = 0
            for p in parasites:
                if beatsums[p] > 0:
                    hsum += self._beats(h, p) * 1./beatsums[p]
            fitnesses.append(hsum)        
        return fitnesses
    
    def _evaluatePopulation(self):
        hoFtournSize = min(self.generation, int(self.tournamentSize * self.hallOfFameEvaluation))
        tournSize = self.tournamentSize - hoFtournSize
        if self.useSharedSampling and self.generation > 2:
            opponents = self._sharedSampling(tournSize, self.parasitePop, self.oldPops[-2])
        else:
            opponents = self.parasitePop
        if len(opponents) < tournSize:
            tournSize = len(opponents)
        self._doTournament(self.pop, opponents, tournSize)
        if hoFtournSize > 0:
            self._doTournament(self.pop, self.hallOfFame, hoFtournSize)
        return self._competitiveSharedFitness(self.pop, self.parasitePop)
    
    def _oneGeneration(self):
        Coevolution._oneGeneration(self)
        # change roles between parasites and hosts
        tmp = self.pop
        self.pop = self.parasitePop
        self.parasitePop = tmp
        
        