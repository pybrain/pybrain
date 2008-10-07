__author__ = 'Daan Wierstra and Tom Schaul'

from scipy import dot, rand, ones, eye, zeros, outer, isnan, multiply
from numpy.random import multivariate_normal
from numpy import average

from pybrain.utilities import drawIndex
from blackboxoptimizer import BlackBoxOptimizer
from pybrain.tools.rankingfunctions import RankingFunction
from pybrain.tools.functions import multivariateNormalPdf, multivariateCauchy


class FEM(BlackBoxOptimizer):
    """ Fitness expectation-maximization"""
    
    batchsize = 50 #a.k.a: lambda
    numberOfCenters = 1  #a.k.a: k
    rangemins = None
    rangemaxs = None
    initCovariances = None
    onlineLearning = True
    forgetFactor = 0.5
    elitist = False
    evalMus = True
    rankingFunction = RankingFunction()
    useCauchy = False
    
    # TODO: interface changed: make coherent
    online = False
    
    def __init__(self, evaluator, evaluable, **parameters):
        BlackBoxOptimizer.__init__(self, evaluator, evaluable, **parameters)
        self.alphas = ones(self.numberOfCenters)/self.numberOfCenters
        self.mus = []
        self.sigmas = []

        self.tau = 1.
        if self.rangemins == None:
            self.rangemins = -ones(self.xdim)
        if self.rangemaxs == None:
            self.rangemaxs = ones(self.xdim)
        if self.initCovariances == None:
            self.initCovariances = eye(self.xdim)
            
        if self.elitist and self.numberOfCenters == 1 and not self.noisyEvaluator:
            # in the elitist case seperate evaluations are not necessary. 
            # CHECKME: maybe in the noisy case?
            self.evalMus = False
            
        assert not(self.useCauchy and self.numberOfCenters > 1)
            
        for dummy in range(self.numberOfCenters):
            self.mus.append(rand(self.xdim) * (self.rangemaxs-self.rangemins) + self.rangemins)
            self.sigmas.append(dot(eye(self.xdim), self.initCovariances))
        self.reset()
            
    def reset(self):
        self.samples = range(self.batchsize)
        self.densities = zeros((self.batchsize, self.numberOfCenters))
        self.fitnesses = zeros(self.batchsize)
        self.generation = 0
        self.allsamples = []
        self.muevals = []
        self.allsigmas =[]
        
    def _stoppingCriterion(self):
        if self.evalMus:
            evals = len(self.allsamples)+len(self.muevals)
        else:
            evals = len(self.allsamples)
        return (self.bestEvaluation >= self.desiredEvaluation or evals >= self.maxEvaluations)

    def _batchLearn(self, maxSteps):
        if self.verbose:
            print
            print "==================="        
            print "Fitness Expectation Maximization"
            print "==================="
            if self.onlineLearning:
                print "ONLINE"
                print "Forget-factor:", self.forgetFactor
            else:
                print 'OFFLINE'
            print 'Distribution:',
            if self.useCauchy:
                print 'Cauchy'
            else:
                print 'Gaussian'
            print "Batch-size:", self.batchsize
            print "Elitist:", self.elitist
            print 'Ranking function:', self.rankingFunction.name
            if self.numberOfCenters > 1:
                print "Number of centers:", self.numberOfCenters
            print
        
        # go through a number of generations
        while not self._stoppingCriterion():
            for k in range(self.batchsize):
                self.samples[k], self.fitnesses[k], self.densities[k] = self._produceNewSample()
                
                if self.onlineLearning and self.generation >= 1:
                    self._updateWeightings()
                    self._updateParameters(k)
                    
                if self._stoppingCriterion(): break
                
            if not self.onlineLearning:
                self._updateWeightings()
                self._updateParameters()
                                            
            # evaluate the mu points seperately (for filtered progression values)
            if self.evalMus:
                for m in self.mus:
                    me = self.evaluator(m)
                    if me > self.bestEvaluation:
                        self.bestEvaluation, self.bestEvaluable = me, m
                    self.muevals.append(me)
                    import copy
                    self.allsigmas.append(copy.deepcopy(self.sigmas))
            else:
                self.muevals.append(self.bestEvaluation)
                    
            if self.verbose:
                print 'gen:', self.generation, 'max,min,avg:',max(self.fitnesses), min(self.fitnesses), average(self.fitnesses),
                if self.evalMus: print '   mu-fitness(es):', self.muevals[-len(self.mus):]
                else: print
            
            self.generation += 1
            self.notify()            
                                 
    def _produceNewSample(self):
        """ returns a new sample, its fitness and its densities """
        sample = self._generateSample()
        fit = self.evaluator(sample)
        if fit >= self.bestEvaluation:
            self.bestEvaluation = fit
            self.bestEvaluable = sample.copy()
        self.allsamples.append(sample)
        
        # compute densities, and normalize
        densities = zeros(self.numberOfCenters)
        if self.numberOfCenters > 1:
            for c in range(self.numberOfCenters):
                densities[c] = self.alphas[c] * multivariateNormalPdf(sample, self.mus[c], self.sigmas[c])
            densities /= sum(densities)
        
        return sample, fit, densities

    def _generateSample(self):
        """ generate a new sample from the current distribution. """
        if self.useCauchy:
            # Cauchy distribution
            chosenOne = drawIndex(self.alphas, True)
            return multivariateCauchy(self.mus[chosenOne], self.sigmas[chosenOne])
        else:
            # Normal distribution
            chosenOne = drawIndex(self.alphas, True)
            return multivariate_normal(self.mus[chosenOne], self.sigmas[chosenOne])
                    
    def _updateWeightings(self):
        """ update the weightings using transformed fitnesses """
        # determine (transformed) fitnesses
        transformedfitnesses = self.rankingFunction(self.fitnesses)
        # force renormaliziation
        transformedfitnesses /= max(transformedfitnesses)
        
        if self.numberOfCenters > 1:    
            self.weightings = multiply(outer(transformedfitnesses, ones(self.numberOfCenters)), self.densities)
        else:
            self.weightings = transformedfitnesses.reshape(self.batchsize, 1)
        
        if self.onlineLearning:            
            for c in range(self.numberOfCenters):
                self.weightings[:,c] /= max(self.weightings[:,c])
        else:
            #CHECKME: inconsistency?
            self.weightings /= sum(self.weightings)     
        
    def _cauchyUpdate(self, weightings):
        """ computation of parameter updates if the distribution is Cauchy """
        # make sure the weightings sum to 1
        weightings  = weightings / sum(weightings)
        newSigma = zeros((self.xdim, self.xdim))                
        newMu = zeros(self.xdim)
        for d in range(self.xdim):
            # build a list of tuples of (value, weight)
            tuplist = zip(map(lambda s: s[d], self.samples), weightings)
            tuplist.sort()
            # determine the values at the 1/4 and 3/4 points of cumulative weighting
            cum = 0
            quart = None
            for elem, w in tuplist:
                cum += w
                if cum >= 0.25 and not quart:
                    quart = elem
                if cum >= 0.75:
                    threequart = elem
                    break
            assert threequart != quart                    
            newMu[d] = (quart + threequart)/2
            newSigma[d,d] = threequart - newMu[d]
        return newMu, newSigma
    
    def _gaussianUpdate(self, weightings, indices, oldMu):
        """ computation of parameter updates if the distribution is gaussian """
        newMu = zeros(self.xdim)
        newSigma = zeros((self.xdim, self.xdim))
        for i in indices:
            newMu += weightings[i] * self.samples[i]
        for i in indices:
            dif = self.samples[i]-newMu
            newSigma += weightings[i] * outer(dif, dif) 
        return newMu, newSigma
        
    def _updateParameters(self, sampleIndex = None):
        for c in range(self.numberOfCenters):
            weightings = self.weightings[:,c]
            if self.onlineLearning:
                lr = self.forgetFactor * weightings[sampleIndex]
                self.alphas[c] = (1.0-lr) * self.alphas[c] + lr            
            else:
                self.alphas[c] = sum(weightings)
                
            # determine the updates to the parameters, depending on the distribution used
            if self.useCauchy:
                newMu, newSigma = self._cauchyUpdate(weightings)
            else:
                # gaussian case
                if self.onlineLearning:
                    newMu, newSigma = self._gaussianUpdate(weightings, [sampleIndex], self.mus[c])
                else:
                    newMu, newSigma = self._gaussianUpdate(weightings, range(self.batchsize), self.mus[c],)
                    # CHECKME: redundant!?
                    #newMu /= sum(weightings)
                    #newSigma /= sum(weightings)
    
            # update the mus
            if self.elitist:
                self.mus[c] = self.bestEvaluable.copy()
            else:
                if self.onlineLearning:
                    # use the forget-factor
                    self.mus[c] = (1-lr) * self.mus[c] + self.forgetFactor * newMu
                else:
                    self.mus[c] = newMu
            
            # update the sigmas
            if True in isnan(newSigma):
                print "NaNs! We'll ignore them."
            else: 
                if self.onlineLearning:
                    # use the forget-factor
                    self.sigmas[c] = (1-lr) * self.sigmas[c] + self.forgetFactor * newSigma
                else:
                    self.sigmas[c] = newSigma 
            
        # nomalize alphas
        self.alphas /= sum(self.alphas)
    