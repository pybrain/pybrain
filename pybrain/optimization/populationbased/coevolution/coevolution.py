from __future__ import print_function

__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import argmax, array
from random import sample, choice, shuffle

from pybrain.utilities import fListToString, Named


class Coevolution(Named):
    """ Population-based generational evolutionary algorithm
    with fitness being based (paritally) on a relative measure. """

    # algorithm parameters
    populationSize = 50
    selectionProportion = 0.5
    elitism = False
    parentChildAverage = 1. # proportion of the child
    tournamentSize = None
    hallOfFameEvaluation = 0. # proportion of HoF evaluations in relative fitness
    useSharedSampling = False

    # an external absolute evaluator
    absEvaluator = None
    absEvalProportion = 0

    # execution settings
    maxGenerations = None
    maxEvaluations = None
    verbose = False

    def __init__(self, relEvaluator, seeds, **args):
        """
        :arg relevaluator: an anti-symmetric function that can evaluate 2 elements
        :arg seeds: a list of initial guesses
        """
        # set parameters
        self.setArgs(**args)
        self.relEvaluator = relEvaluator
        if self.tournamentSize == None:
            self.tournamentSize = self.populationSize

        # initialize algorithm variables
        self.steps = 0
        self.generation = 0

        # the best host and the best parasite from each generation
        self.hallOfFame = []

        # the relative fitnesses from each generation (of the selected individuals)
        self.hallOfFitnesses = []

        # this dictionary stores all the results between 2 players (first one starting):
        #  { (player1, player2): [games won, total games, cumulative score, list of scores] }
        self.allResults = {}

        # this dictionary stores the opponents a player has played against.
        self.allOpponents = {}

        # a list of all previous populations
        self.oldPops = []

        # build initial populations
        self._initPopulation(seeds)

    def learn(self, maxSteps=None):
        """ Toplevel function, can be called iteratively.

        :return: best evaluable found in the last generation. """
        if maxSteps != None:
            maxSteps += self.steps
        while True:
            if maxSteps != None and self.steps + self._stepsPerGeneration() > maxSteps:
                break
            if self.maxEvaluations != None and self.steps + self._stepsPerGeneration() > self.maxEvaluations:
                break
            if self.maxGenerations != None and self.generation >= self.maxGenerations:
                break
            self._oneGeneration()
        return self.hallOfFame[-1]

    def _oneGeneration(self):
        self.oldPops.append(self.pop)
        self.generation += 1
        fitnesses = self._evaluatePopulation()
        # store best in hall of fame
        besti = argmax(array(fitnesses))
        best = self.pop[besti]
        bestFits = sorted(fitnesses)[::-1][:self._numSelected()]
        self.hallOfFame.append(best)
        self.hallOfFitnesses.append(bestFits)

        if self.verbose:
            print(('Generation', self.generation))
            print(('        relat. fits:', fListToString(sorted(fitnesses), 4)))
            if len(best.params) < 20:
                print(('        best params:', fListToString(best.params, 4)))

        self.pop = self._selectAndReproduce(self.pop, fitnesses)

    def _averageWithParents(self, pop, childportion):
        for i, p in enumerate(pop[:]):
            if p.parent != None:
                tmp = p.copy()
                tmp.parent = p.parent
                tmp._setParameters(p.params * childportion + p.parent.params * (1 - childportion))
                pop[i] = tmp

    def _evaluatePopulation(self):
        hoFtournSize = min(self.generation, int(self.tournamentSize * self.hallOfFameEvaluation))
        tournSize = self.tournamentSize - hoFtournSize
        if self.useSharedSampling:
            opponents = self._sharedSampling(tournSize, self.pop, self.oldPops[-1])
        else:
            opponents = self.pop
        if len(opponents) < tournSize:
            tournSize = len(opponents)
        self._doTournament(self.pop, opponents, tournSize)
        if hoFtournSize > 0:
            hoF = list(set(self.hallOfFame))
            self._doTournament(self.pop, hoF, hoFtournSize)
        fitnesses = []
        for p in self.pop:
            fit = 0
            for opp in opponents:
                fit += self._beats(p, opp)
            if hoFtournSize > 0:
                for opp in hoF:
                    fit += self._beats(p, opp)
            if self.absEvalProportion > 0 and self.absEvaluator != None:
                fit = (1 - self.absEvalProportion) * fit + self.absEvalProportion * self.absEvaluator(p)
            fitnesses.append(fit)
        return fitnesses

    def _initPopulation(self, seeds):
        if self.parentChildAverage < 1:
            for s in seeds:
                s.parent = None
        self.pop = self._extendPopulation(seeds, self.populationSize)

    def _extendPopulation(self, seeds, size):
        """ build a population, with mutated copies from the provided
        seed pool until it has the desired size. """
        res = seeds[:]
        for dummy in range(size - len(seeds)):
            chosen = choice(seeds)
            tmp = chosen.copy()
            tmp.mutate()
            if self.parentChildAverage < 1:
                tmp.parent = chosen
            res.append(tmp)
        return res

    def _selectAndReproduce(self, pop, fits):
        """ apply selection and reproduction to host population, according to their fitness."""
        # combine population with their fitness, then sort, only by fitness
        s = list(zip(fits, pop))
        shuffle(s)
        s.sort(key=lambda x:-x[0])
        # select...
        selected = [x[1] for x in s[:self._numSelected()]]
        # ... and reproduce
        if self.elitism:
            newpop = self._extendPopulation(selected, self.populationSize)
            if self.parentChildAverage < 1:
                self._averageWithParents(newpop, self.parentChildAverage)
        else:
            newpop = self._extendPopulation(selected, self.populationSize
                                            + self._numSelected()) [self._numSelected():]
            if self.parentChildAverage < 1:
                self._averageWithParents(newpop[self._numSelected():], self.parentChildAverage)
        return newpop

    def _beats(self, h, p):
        """ determine the empirically observed score of p playing opp (starting or not).
        If they never played, assume 0. """
        if (h, p) not in self.allResults:
            return 0
        else:
            hpgames, hscore = self.allResults[(h, p)][1:3]
            phgames, pscore = self.allResults[(p, h)][1:3]
            return (hscore - pscore) / float(hpgames + phgames)

    def _doTournament(self, pop1, pop2, tournamentSize=None):
        """ Play a tournament.

        :key tournamentSize: If unspecified, play all-against-all
        """
        # TODO: Preferably select high-performing opponents?
        for p in pop1:
            pop3 = pop2[:]
            while p in pop3:
                pop3.remove(p)
            if tournamentSize != None and tournamentSize < len(pop3):
                opps = sample(pop3, tournamentSize)
            else:
                opps = pop3
            for opp in opps:
                self._relEval(p, opp)
                self._relEval(opp, p)

    def _globalScore(self, p):
        """ The average score over all evaluations for a player. """
        if p not in self.allOpponents:
            return 0.
        scoresum, played = 0., 0
        for opp in self.allOpponents[p]:
            scoresum += self.allResults[(p, opp)][2]
            played += self.allResults[(p, opp)][1]
            scoresum -= self.allResults[(opp, p)][2]
            played += self.allResults[(opp, p)][1]
        # slightly bias the global score in favor of players with more games (just for tie-breaking)
        played += 0.01
        return scoresum / played

    def _sharedSampling(self, numSelect, selectFrom, relativeTo):
        """ Build a shared sampling set of opponents """
        if numSelect < 1:
            return []
        # determine the player of selectFrom with the most wins against players from relativeTo (and which ones)
        tmp = {}
        for p in selectFrom:
            beaten = []
            for opp in relativeTo:
                if self._beats(p, opp) > 0:
                    beaten.append(opp)
            tmp[p] = beaten
        beatlist = [(len(p_beaten[1]), self._globalScore(p_beaten[0]), p_beaten[0]) for p_beaten in list(tmp.items())]
        shuffle(beatlist)
        beatlist.sort(key=lambda x: x[:2])
        best = beatlist[-1][2]
        unBeaten = list(set(relativeTo).difference(tmp[best]))
        otherSelect = selectFrom[:]
        otherSelect.remove(best)
        return [best] + self._sharedSampling(numSelect - 1, otherSelect, unBeaten)

    def _relEval(self, p, opp):
        """ a single relative evaluation (in one direction) with the involved bookkeeping."""
        if p not in self.allOpponents:
            self.allOpponents[p] = []
        self.allOpponents[p].append(opp)
        if (p, opp) not in self.allResults:
            self.allResults[(p, opp)] = [0, 0, 0., []]
        res = self.relEvaluator(p, opp)
        if res > 0:
            self.allResults[(p, opp)][0] += 1
        self.allResults[(p, opp)][1] += 1
        self.allResults[(p, opp)][2] += res
        self.allResults[(p, opp)][3].append(res)
        self.steps += 1

    def __str__(self):
        s = 'Coevolution ('
        s += str(self._numSelected())
        if self.elitism:
            s += '+' + str(self.populationSize - self._numSelected())
        else:
            s += ',' + str(self.populationSize)
        s += ')'
        if self.parentChildAverage < 1:
            s += ' p_c_avg=' + str(self.parentChildAverage)
        return s

    def _numSelected(self):
        return int(self.populationSize * self.selectionProportion)

    def _stepsPerGeneration(self):
        res = self.populationSize * self.tournamentSize * 2
        return res





if __name__ == '__main__':
    # TODO: convert to unittest
    x = Coevolution(None, [None], populationSize=1)
    x.allResults[(1, 2)] = [1, 1, 1, []]
    x.allResults[(2, 1)] = [-1, 1, -1, []]
    x.allResults[(2, 5)] = [1, 1, 2, []]
    x.allResults[(5, 2)] = [-1, 1, -1, []]
    x.allResults[(2, 3)] = [1, 1, 3, []]
    x.allResults[(3, 2)] = [-1, 1, -1, []]
    x.allResults[(4, 3)] = [1, 1, 4, []]
    x.allResults[(3, 4)] = [-1, 1, -1, []]
    x.allOpponents[1] = [2]
    x.allOpponents[2] = [1, 5]
    x.allOpponents[3] = [2, 4]
    x.allOpponents[4] = [3]
    x.allOpponents[5] = [2]
    print((x._sharedSampling(4, [1, 2, 3, 4, 5], [1, 2, 3, 4, 6, 7, 8, 9])))
    print(('should be', [4, 1, 2, 5]))
