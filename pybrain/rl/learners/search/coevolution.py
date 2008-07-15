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
    
    # an external absolute evaluator
    absEvaluator = None
    absEvalProportion = 0
    
    # execution settings
    maxGenerations = None
    maxEvaluations = None
    verbose = False
    
    def __init__(self, relEvaluator, seeds, **args):
        """ 
        @param relevaluator: an anti-symmetric function that can evaluate 2 elements
        @param seeds: a list of initial guesses
        """
        # set parameters
        self.setArgs(**args)
        self.relEvaluator = relEvaluator
        
        # initialize algorithm variables
        self.steps = 0        
        self.generation = 0
        
        # the best host and the best parasite from each generation
        self.hallOfFame = []
        
        # the relative fitnesses from each generation (of the selected individuals)
        self.hallOfFitnesses = []
        
        # this dictionnary stores all the results between 2 players (first one starting): 
        #  { (player1, player2): [games won, total games] }
        self.allResults = {}
        
        # build initial populations
        self._initPopulation(seeds)
        
    def learn(self, maxSteps = None):
        """ Toplevel function, can be called iteratively.
        @return: best evaluable found in the last generation. """
        if maxSteps != None:
            maxSteps += self.steps
        while True:
            if maxSteps != None and self.steps+self._stepsPerGeneration() > maxSteps:
                break
            if self.maxEvaluations != None and self.steps+self._stepsPerGeneration() > self.maxEvaluations:
                break
            if self.maxGenerations != None and self.generation >= self.maxGenerations:
                break
            self._oneGeneration()
        return self.hallOfFame[-1]

    def _oneGeneration(self):
        self.generation += 1        
        fitnesses = self._evaluatePopulation()
        # store best in hall of fame
        besti = argmax(array(fitnesses))
        best = self.pop[besti]
        bestFits = sorted(fitnesses)[::-1][:self._numSelected()]
        self.hallOfFame.append(best)
        self.hallOfFitnesses.append(bestFits)
                
        if self.verbose:
            print 'Generation', self.generation
            print '        relat. fits:', fListToString(sorted(fitnesses), 4)
            print '        best params:', fListToString(best.params, 4)
                
        self.pop = self._selectAndReproduce(self.pop, fitnesses)
            
        
    def _averageWithParents(self, pop, childportion):
        for i, p in enumerate(pop[:]):
            if p.parent != None:
                tmp = p.copy()
                tmp.parent = p.parent
                tmp._setParameters(p.params * childportion + p.parent.params * (1-childportion))
                pop[i] = tmp
                    
    def _evaluatePopulation(self):
        self._doTournament(self.pop, self.pop, self.tournamentSize)
        if (self.hallOfFameEvaluation > 0 and 
            ((self.tournamentSize != None and self.generation > self.tournamentSize)
             or (self.tournamentSize == None and self.generation > 3))):
            self._doTournament(self.pop, self.hallOfFame, self.tournamentSize)
        fitnesses = []
        for p in self.pop:
            fit = 0
            for opp in self.pop:
                fit += self._beats(p, opp)
            if self.hallOfFameEvaluation > 0:
                for opp in self.hallOfFame:
                    fit += self._beats(p, opp)     
            if self.absEvalProportion > 0 and self.absEvaluator != None:
                fit = (1-self.absEvalProportion) * fit + self.absEvalProportion * self.absEvaluator(p)           
            fitnesses.append(fit)
        return fitnesses
            
    def _initPopulation(self, seeds):
        if self.parentChildAverage < 0:
            for s in seeds:
                s.parent = None
        self.pop = self._extendPopulation(seeds, self.populationSize)
            
    def _extendPopulation(self, seeds, size):
        """ build a population, with mutated copies from the provided
        seed pool until it has the desired size. """
        res = seeds[:]
        for dummy in range(size-len(seeds)):
            chosen = choice(seeds)
            tmp = chosen.copy()
            tmp.mutate()
            if self.parentChildAverage < 0:
                tmp.parent = chosen
            res.append(tmp)            
        return res
        
    def _selectAndReproduce(self, pop, fits):
        """ apply selection and reproduction to host population, according to their fitness."""
        # combine population with their fitness, then sort, only by fitness
        s = zip(fits, pop)
        shuffle(s)
        s.sort(key = lambda x: -x[0])
        # select...
        selected = map(lambda x: x[1], s[:self._numSelected()])
        # ... and reproduce
        if self.elitism:
            newpop = self._extendPopulation(selected, self.populationSize)
            if self.parentChildAverage < 1:
                self._averageWithParents(newpop, self.parentChildAverage)
        else:
            newpop = self._extendPopulation(selected, self.populationSize
                                            +self._numSelected()) [self._numSelected():]
            if self.parentChildAverage < 1:
                self._averageWithParents(newpop[self._numSelected():], self.parentChildAverage)
        return newpop
            
    def _beats(self, h, p):
        """ determine the empirically observed score of p playing opp (starting or not). 
        If they never played, assume 0. """
        if (h,p) not in self.allResults:
            return 0
        else:
            hscore, hpgames = self.allResults[(h,p)]
            pscore, phgames = self.allResults[(p,h)]
            return (hscore-pscore)/float(hpgames+phgames)            
                
    def _doTournament(self, pop1, pop2, tournamentSize = None):
        """ Play a tournament. 
        @param tournamentSize: If unspecified, play all-against-all 
        """
        # TODO: Preferably select high-performing opponents?
        for p in pop1:
            pop3 = pop2[:]
            if p in pop3:
                pop3.remove(p)
            if tournamentSize != None and tournamentSize < len(pop3):                
                opps = sample(pop3, tournamentSize)
            else:                
                opps = pop3                    
            for opp in opps:
                assert p != opp
                if (p,opp) not in self.allResults:
                    self.allResults[(p,opp)] = [0,0]
                if (opp,p) not in self.allResults:
                    self.allResults[(opp,p)] = [0,0]
                self.allResults[(p,opp)][1] += 1
                self.allResults[(p,opp)][0] += self.relEvaluator(p, opp)
                self.allResults[(opp,p)][1] += 1
                self.allResults[(opp,p)][0] += self.relEvaluator(opp, p)
                self.steps += 2     
    
    def __str__(self):
        s = 'Coevolution ('
        s += str(self._numSelected())
        if self.elitism:
            s += '+'+str(self.populationSize-self._numSelected())
        else:
            s += ','+str(self.populationSize)
        s += ')'
        if self.parentChildAverage < 1:
             s += ' p_c_avg='+str(self.parentChildAverage)
        return s
                                     
    def _numSelected(self):
        return int(self.populationSize*self.selectionProportion)
    
    def _stepsPerGeneration(self):
        if self.tournamentSize == None:
            res = self.populationSize*(self.populationSize-1) * 2
        else:
            res = self.populationSize*self.tournamentSize * 2
        if self.hallOfFameEvaluation > 0:
            res *= 2
        return res