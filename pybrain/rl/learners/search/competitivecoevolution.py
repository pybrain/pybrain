__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import argmax, array
from random import shuffle

from pybrain.utilities import Named


class CompetitiveCoevolution(Named):
    """ 2-population competitive coevolution, following the host-parasite paradigm. """
    
    maxGenerations = None
    maxEvaluations = None
    populationSize = 50
    selectionProportion = 0.5
    elitism = False
    verbose = False
    
    # proportion of the parent's weights intermixed with offspring
    # in case of successful children
    parentChildAvg = 0.
    
    def __init__(self, evaluator, evaluable1, evaluable2, **args):
        self.setArgs(**args)
        self.evaluator = evaluator
        self.steps = 0
        self.generation = 0
        # all evaluables get an id = gen+x
        self.allEvaluables = {}
        # build initial populations
        self.hostPop = self._initPopulation(evaluable1)
        self.parasitePop = self._initPopulation(evaluable2)
        # the best host and the best parasite from each generation
        self.hallOfFame = []
        # this dictionnary stores all the results between 2 players (first one starting): 
        #  { (player1.id, player2.id): [games won, total games] }
        self.allResults = {}
                
    def learn(self, maxSteps = None):
        """ @return: (best evaluable found, best fitness) """
        if maxSteps != None:
            maxSteps += self.steps
        while True:
            if maxSteps != None and self.steps+self._stepsPerGeneration() > maxSteps:
                break
            if self.maxEvaluations != None and self.steps+self._stepsPerGeneration() > self.maxEvaluations:
                break
            if self.maxGenerations != None and self.generation >= self.maxGenerations:
                break
            self.oneGeneration()
        return self.hallOfFame[-1]
                                                        
    def oneGeneration(self):
        self.generation += 1
        self._doTournament(self.hostPop, self.parasitePop)
        # determine beat-sum for parasites (nb of games lost)
        beatsums = {}
        for p in self.parasitePop:
            beatsums[p.id] = 0
            for h in self.hostPop:
                beatsums[p.id] += self._beats(h, p)
                
        # determine fitnesses for hosts
        fitnesses = []
        for h in self.hostPop:
            hsum = 0
            for p in self.parasitePop:
                if beatsums[p.id] > 0:
                    hsum += self._beats(h, p) * 1./beatsums[p.id]
            fitnesses.append(hsum)        
        
        if self.verbose:
            print 'Generation', self.generation
            from pybrain.utilities import fListToString
            print fListToString(fitnesses, 4)
                
        # store best host in hall of fame
        self.hallOfFame.append(self.hostPop[argmax(array(fitnesses))])
        
        # evolution in the host population
        self.hostPop = self._evolvePopulation(self.hostPop, fitnesses)
        
        # change roles between parasites and hosts
        tmp = self.hostPop
        self.hostPop = self.parasitePop
        self.parasitePop = tmp            
                
    def _evolvePopulation(self, pop, fits):
        """ apply selection and reproduction to host population, according to their fitness."""
        # combine population with their fitness, then sort 
        s = zip(fits, pop)
        shuffle(s)
        s.sort(key = lambda x: -x[0])
                
        selected = int(self.populationSize*self.selectionProportion)
        if self.elitism:
            # copy the best part
            res = map(lambda x: x[1], s)[:selected]
        else:
            res = []
        
        while True:
            for i in range(selected):
                tmp = s[i][1].copy()
                tmp.mutate()
                id = str(self.generation)+'-'+str(self._nextId())
                self.allEvaluables[id] = tmp
                tmp.id = id
                res.append(tmp)
                if len(res) == self.populationSize:
                    return res
        
    def _beats(self, h, p):
        """ determine the empirically observed probability of h beating p (staring or not). """
        hwins, hpgames = self.allResults[(h.id,p.id)]
        pwins, phgames = self.allResults[(p.id,h.id)]
        return (hwins+(phgames-pwins))/float(hpgames+phgames)            
        
    def _initPopulation(self, seed):
        res = [seed]
        seed.id = 'seed-'+str(self._nextId())
        for dummy in range(self.populationSize-1):
            tmp = seed.copy()
            tmp.mutate()
            res.append(tmp)
            id = str(self.generation)+'-'+str(self._nextId())
            self.allEvaluables[id] = tmp
            tmp.id = id
        return res
        
    def _doTournament(self, pop1, pop2):
        """ All hosts play 2 games against all parasites (one as first player, one as second). """
        for h in pop1:
            for p in pop2:
                if h == p:
                    continue
                if (h.id,p.id) not in self.allResults:
                    self.allResults[(h.id,p.id)] = [0,0]
                self.allResults[(h.id,p.id)][1] += 1
                hwin = self.evaluator(h, p)
                if hwin:
                    self.allResults[(h.id,p.id)][0] += 1
                
                if (p.id,h.id) not in self.allResults:
                    self.allResults[(p.id,h.id)] = [0,0]
                self.allResults[(p.id,h.id)][1] += 1
                pwin = self.evaluator(p, h)
                if pwin:
                    self.allResults[(p.id,h.id)][0] += 1
                self.steps += 2
                
    def _stepsPerGeneration(self):
        return self.populationSize**2
    
    def _nextId(self, x = [0]):
        x[0] += 1
        return x[0]