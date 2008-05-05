__author__ = 'Daan Wierstra and Tom Schaul'

from scipy import dot, rand, ones, eye, zeros, outer, isnan
from numpy.random import multivariate_normal
from numpy import average

from pybrain.utilities import drawIndex
from blackboxoptimizer import BlackBoxOptimizer
from pybrain.tools.rankingfunctions import RankingFunction
from pybrain.tools.functions import multivariateNormalPdf



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
                    self._updateGaussianParameters(k)
                    
                if self._stoppingCriterion(): break
                
            if not self.onlineLearning:
                self._updateWeightings()
                self._batchUpdateGaussianParameters()
            
                                
            # evaluate the mu points seperately (for filtered progression values)
            if self.evalMus:
                for m in self.mus:
                    me = self.evaluator(m)
                    if me > self.bestEvaluation:
                        self.bestEvaluation, self.bestEvaluable = me, m
                    self.muevals.append(me)
            else:
                self.muevals.append(self.bestEvaluation)
                    
            if self.verbose:
                print 'gen:', self.generation, 'max,min,avg:',max(self.fitnesses), min(self.fitnesses), average(self.fitnesses),
                if self.evalMus: print '   mu-fitness(es):', self.muevals[-len(self.mus):]
                else: print
            
            self.generation += 1             
                                 
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
        for c in range(self.numberOfCenters):
            densities[c] = self.alphas[c] * multivariateNormalPdf(sample, self.mus[c], self.sigmas[c])
        densities /= sum(densities)
        
        return sample, fit, densities

    def _generateSample(self):
        """ generate a new sample from the current distribution. """
        
        # gaussian case:
        chosenOne = drawIndex(self.alphas, True)
        return multivariate_normal(self.mus[chosenOne], self.sigmas[chosenOne])
                    
    def _updateWeightings(self):
        """ update the weightings using transformed fitnesses """
        # determine (transformed) fitnesses
        transformedfitnesses = self.rankingFunction(self.fitnesses)
        # force renormaliziation
        transformedfitnesses /= max(transformedfitnesses)
        
        # CHECKME: using densities?
        # self.weightings = multiply(outer(transformedfitnesses, ones(self.numberOfCenters)), self.densities)
        self.weightings = outer(transformedfitnesses, ones(self.numberOfCenters))
        self.weightings /= max(self.weightings)

    
    def _updateGaussianParameters(self, sampleindex):
        """ update the mu(s), sigma(s) (and alphas), using the current weightings,
        but only on the specified sample-index, using a forget rate. """
        for c in range(self.numberOfCenters):
            lr = self.forgetFactor * self.weightings[sampleindex,c]
                                
            self.alphas[c] = (1.0-lr) * self.alphas[c] + lr
            
            if self.elitist:
                self.mus[c] = self.bestEvaluable.copy()
            else:
                self.mus[c] = (1.0-lr) * self.mus[c] + lr * self.samples[sampleindex]

            dif = self.samples[sampleindex]-self.mus[c]
            newSigma = (1.0-lr) * self.sigmas[c] + lr * outer(dif, dif) 
            if True in isnan(newSigma):
                print "NaNs! We'll ignore them."
            else: 
                self.sigmas[c] = newSigma 
                    
        # nomalize alphas
        self.alphas /= sum(self.alphas)
        
    def _batchUpdateGaussianParameters(self):
        """ update the mu(s), sigma(s) (and alphas), using the current weightings """
        for c in range(self.numberOfCenters):
            self.alphas[c] = sum(self.weightings[:,c])
            
            if self.elitist:
                self.mus[c] = self.bestEvaluable.copy()
            else:
                newMu = zeros(self.xdim)
                for i in range(self.batchsize):
                    newMu += self.weightings[i,c] * self.samples[i]
                newMu /= sum(self.weightings[:,c])
                self.mus[c] = newMu
                
            newSigma = zeros((self.xdim, self.xdim))
            for i in range(self.batchsize):
                dif = -self.mus[c]+self.samples[i]
                newSigma += self.weightings[i,c] * outer(dif, dif) 
            newSigma /= sum(self.weightings[:,c])
            if True in isnan(newSigma):
                print "NaNs! We'll ignore them."
            else: 
                self.sigmas[c] = newSigma 
                    
        # nomalize alphas
        self.alphas /= sum(self.alphas)
        
        