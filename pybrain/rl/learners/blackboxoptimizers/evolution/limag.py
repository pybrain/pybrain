__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import zeros, ones, mean, array, argmax
from random import choice, shuffle

from ga import GA
from pybrain.tools.functions import sigmoid
from pybrain.utilities import drawIndex


class LiMaG(GA):
    """ Linkage Matrix gradients """

    learningRate = 0.001
    popsize = 20
    topproportion = .5
    mutationStdDev = 0.01
    
    verbose = False
    
    useMatrixForCrossover = False
    
    multiParents = True
    
    fitnessSmoothing = False
    
    def __init__(self, f, **args):
        GA.__init__(self, f, **args)
        self.initLinkageMatrix()
        self.averageFitness = 0
        self.parentFitnesses = None
        
    def initLinkageMatrix(self):
        """ initialize the matrix to a uniformly uncorrelated gene-picking from any selected parent. """
        self.rawlm = ones((self.xdim, self.xdim))
        if self.multiParents:
            self.rawlm /= self.selectionSize()
        else:
            self.rawlm /= 2
        self._transformLinkages()
    
    def crossOver(self, parents, nbChildren):
        """ do the crossovers according to the linkage matrix, but keep track of the 
        crossover vectors (which gene was picked from which parent). """
        children = []
        self.crossovervectors = []
        for dummy in range(nbChildren):
            if self.useMatrixForCrossover:
                child, crossovervector = self._generateOneOffspring(parents)
            else:
                child, crossovervector = self._uniformCrossover(parents)
            children.append(child)
            self.crossovervectors.append(crossovervector)
        return children     
    
    def oneGeneration(self):
        if self.generation > 0:
            self.calculateAverageFitness()
        # evaluate fitness
        self.fitnesses = []
        for indiv in self.currentpop:
            self.fitnesses.append(self.targetfun(indiv))
        
        # determine the best values
        best = argmax(array(self.fitnesses))
        self.bestfitness = self.fitnesses[best]
        self.bestx = self.currentpop[best]
        
        self.allgenerations.append((self.currentpop, self.fitnesses))
        
        if self.fitnessSmoothing:
            self._smoothFitnesses()    
        
        # selection
        tmp = zip(self.fitnesses, self.currentpop)
        tmp.sort(key = lambda x: x[0])            
        tmp2 = list(reversed(tmp))[:self.selectionSize()]
        parents, self.parentFitnesses = map(lambda x: x[1], tmp2), map(lambda x: x[0], tmp2)
        
        self.currentpop = self.crossOver(parents, self.popsize)
        
        # add one random offspring
        #self.currentpop[-1] = randn(self.xdim)
        #self.crossovervectors[-1] = [0]*self.xdim
        
        for child in self.currentpop:
            self.mutate(child)
        
        if self.generation > 0:
            self.updateLinkageMatrix()
        if self.verbose:# and self.generation % 10 == 0:
            # TODO: more extensive output
            print self.rawlm
        
                    
    def calculateAverageFitness(self):
        self.averageFitness = mean(self.fitnesses)
        
    def _smoothFitnesses(self):
        """do a non-linear, ranking based fitness normalization. """
        if not hasattr(self, 'nes'):
            from pybrain.rl.learners.blackboxoptimizers.nes import NaturalEvolutionStrategies
            self.nes = NaturalEvolutionStrategies(self.tfun)
            self.nes.lambd = self.popsize
        self.fitnesses = self.nes.smoothSelectiveRanking(self.fitnesses)
        
    def updateLinkageMatrix(self):
        """ do the gradient update on the linkage matrix """
        #update untransformed matrix
        for index, crossovervector in enumerate(self.crossovervectors):
            fitness = self.fitnesses[index]         
            baseline = self.averageFitness
            if not self.multiParents:
                p1 = int(min(crossovervector))
                p2 = int(max(crossovervector))
                if p1 == p2: 
                    #CHECKME
                    continue
                baseline = (self.parentFitnesses[p1]+self.parentFitnesses[p2])/2
            if fitness - baseline < 0:
                # lower updates for bad offspring
                baseline = fitness - (fitness-baseline) /10#/(self.selectionSize() - 1)
            if fitness == baseline:
                continue
            
            if self.verbose: 
                print fitness, baseline, crossovervector
            for i in xrange(1,self.xdim):               
                for j in xrange (i):
                    
                    if self.useMatrixForCrossover:
                        lprob = self.lm[i, j]
                    else:
                        if self.multiParents:
                            lprob = 1./self.selectionSize()
                        else:
                            lprob = 0.5

                    if (crossovervector[i] == crossovervector[j]):                    
                        update = (fitness - baseline) * (1 - lprob) #*  4/3                                            
                    else:
                        update = (fitness - baseline) * (0 - lprob) #/ (self.selectionSize() - 1)                
                    self.rawlm[i, j] += update * self.learningRate
                    if self.verbose:
                        print '', i, j, update * self.learningRate
                    
        self._transformLinkages()
                    
    def _transformLinkages(self):
        #print "before", self.lm
        self.lm = sigmoid(self.rawlm)                        
        #print "after", self.lm
                        
    def _generateOneOffspring(self, pop):
        """ produce one single offspring, given the population and the linkage matrix """
        # TODO: optimize?
        n = self.xdim
        # one gene is chosen directly
        initindex = choice(range(n))
        chosen = [(choice(range(len(pop))), initindex)]
        
        # the indices of the rest are shuffled
        indices = list(range(n))
        shuffle(indices)
        indices.remove(initindex)
        
        for index in indices:
            probs = zeros(len(pop))
            for parent in range(len(pop)):
                # determine the probability of drawing the i'th gene from parent p
                p1 = self._computeProbChosenGivenAq(len(pop), index, parent, chosen)
                p2 = self._computeProbChosenGivenAq(len(pop), index, parent, chosen, invertAq = True)   
                probs[parent] = p1 / (p1 + (len(pop)-1)* p2)
            # draw according to the probabilities
            chosen.append((drawIndex(probs, tolerant = True), index))
            
        child = zeros(self.xdim)
        crossovervector = zeros(self.xdim)
        for parent, index in chosen:
            child[index] = pop[parent][index]   
            crossovervector[index] = parent       
        return child, crossovervector 
    
    def _computeProbChosenGivenAq(self, popsize, indexq, parentq, chosen, invertAq = False):
        """ produce the probability of picking gene indexq from parentq given the chosen (parent, gene) tuples. 
        @param invertAq: if this flag is true, the probability is the one of picking the gene NOT from parentq 
        """
        res = 1
        for parentj, indexj in chosen:
            linkage = self.lm[max(indexq, indexj), min(indexq, indexj)]        
            if parentj == parentq:
                if invertAq:
                    res *= (1-linkage)/(popsize-1.)
                else:
                    res *= linkage
            else:
                if invertAq:
                    res *= (popsize-2 + linkage)/(popsize-1)**2
                else:
                    res *= (1-linkage)/(popsize-1.)
        return res
        
    def _uniformCrossover(self, parents):
        """Uniform crossover between all parents. Does not take the linkage matrix into account at all."""
        if not self.multiParents:
            parent1 = choice(range(len(parents)))
            parent2 = choice(range(len(parents)))
        child = zeros(self.xdim)
        crossovervector = zeros(self.xdim)
        for i in range(self.xdim):
            if not self.multiParents:
                p = choice([parent1, parent2])
            else:
                p = choice(range(len(parents)))
            crossovervector[i] = (p)
            child[i] = (parents[p][i])
        return child, crossovervector