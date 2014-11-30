from __future__ import print_function

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.optimization.coevolution.coevolution import Coevolution


class CompetitiveCoevolution(Coevolution):
    """ Coevolution with 2 independent populations, and competitive fitness sharing. """

    def __str__(self):
        return 'Competitive' + Coevolution.__str__(self)

    def _initPopulation(self, seeds):
        """ one half for each population """
        if self.parentChildAverage < 1:
            for s in seeds:
                s.parent = None
        if len(seeds) > 1:
            s1 = seeds[:len(seeds) // 2]
            s2 = seeds[len(seeds) // 2:]
        else:
            # not enough seeds: randomize
            s1 = seeds
            tmp = seeds[0].copy()
            tmp.randomize()
            s2 = [tmp]
        self.pop = self._extendPopulation(s1, self.populationSize)
        self.parasitePop = self._extendPopulation(s2, self.populationSize)

    def _competitiveSharedFitness(self, hosts, parasites):
        """ determine the competitive shared fitness for the population of hosts, w.r. to
        the population of parasites. """
        if len(parasites) == 0:
            return [0] * len(hosts)

        # determine beat-sum for parasites (nb of games lost)
        beatsums = {}
        for p in parasites:
            beatsums[p] = 0.
            for h in hosts:
                if self._beats(h, p) > 0:
                    beatsums[p] += 1

        # determine fitnesses for hosts
        fitnesses = []
        for h in hosts:
            hsum = 0
            unplayed = 0
            for p in parasites:
                if self._beats(h, p) > 0:
                    assert beatsums[p] > 0
                    hsum += 1. / beatsums[p]
                elif self._beats(h, p) == 0:
                    unplayed += 1
            # take into account the number of parasites played, to avoid
            # biasing for old agents in the elitist case
            if len(parasites) > unplayed:
                hsum /= float(len(parasites) - unplayed)

            # this is purely for breaking ties in favor of globally better players:
            hsum += 1e-5 * self._globalScore(h)
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

        fit = self._competitiveSharedFitness(self.pop, self.parasitePop)
        if hoFtournSize > 0:
            fitHof = self._competitiveSharedFitness(self.pop, self.hallOfFame)
            fit = [tournSize * f1_f2[0] + hoFtournSize * f1_f2[1] for f1_f2 in zip(fit, fitHof)]
        return fit

    def _oneGeneration(self):
        Coevolution._oneGeneration(self)
        # change roles between parasites and hosts
        tmp = self.pop
        self.pop = self.parasitePop
        self.parasitePop = tmp






if __name__ == '__main__':
    from pybrain.utilities import fListToString
    # TODO: convert to unittest
    C = CompetitiveCoevolution(None, [1, 2, 3, 4, 5, 6, 7, 8], populationSize=4)
    def b(x, y):
        C.allResults[(x, y)] = [1, 1, 1, []]
        C.allResults[(y, x)] = [-1, 1, -1, []]
        if x not in C.allOpponents:
            C.allOpponents[x] = []
        if y not in C.allOpponents:
            C.allOpponents[y] = []
        C.allOpponents[x].append(y)
        C.allOpponents[y].append(x)

    b(1, 6)
    b(1, 7)
    b(8, 1)
    b(5, 2)
    b(6, 2)
    b(8, 2)
    b(3, 5)
    b(3, 6)
    b(3, 7)
    b(4, 5)
    b(4, 7)
    b(8, 4)
    print((C.pop))
    print((C.parasitePop))
    print(('          ', fListToString(C._competitiveSharedFitness(C.pop, C.parasitePop), 2)))
    print(('should be:', fListToString([0.83, 0.00, 1.33, 0.83], 2)))

