__author__ = 'Daan Wierstra and Tom Schaul'

import copy
from scipy import dot, rand, ones, eye, zeros, outer, isnan, multiply
from numpy.random import multivariate_normal

from pybrain.utilities import drawIndex
from pybrain.tools.rankingfunctions import RankingFunction
from pybrain.tools.functions import multivariateNormalPdf, multivariateCauchy
from pybrain.optimization.optimizer import ContinuousOptimizer


class FEM(ContinuousOptimizer):
    """ Fitness expectation-maximization"""
    
    minimize = True
    
    onlineLearning = True
    batchsize = 50 #a.k.a: lambda
    forgetFactor = 0.5
    elitist = False
    rankingFunction = RankingFunction()
    useCauchy = False
    numberOfCenters = 1  #a.k.a: k    
    
    rangemins = None
    rangemaxs = None
    initCovariances = None

    storeAllCenters = True
    storeAllDistributions = False
    
    def __init__(self, evaluator, evaluable, **parameters):
        ContinuousOptimizer.__init__(self, evaluator, evaluable, **parameters)
        self.alphas = ones(self.numberOfCenters)/self.numberOfCenters
        self.mus = []
        self.sigmas = []

        xdim = self.numParameters
        if self.rangemins == None:
            self.rangemins = -ones(xdim)
        if self.rangemaxs == None:
            self.rangemaxs = ones(xdim)
        if self.initCovariances == None:
            self.initCovariances = eye(xdim)        
        assert not(self.useCauchy and self.numberOfCenters > 1)
            
        for _ in range(self.numberOfCenters):
            self.mus.append(rand(xdim) * (self.rangemaxs-self.rangemins) + self.rangemins)
            self.sigmas.append(dot(eye(xdim), self.initCovariances))
        self.reset()
            
    def reset(self):
        self.samples = range(self.batchsize)
        self.densities = zeros((self.batchsize, self.numberOfCenters))
        self.fitnesses = zeros(self.batchsize)
        self._allCenters = []
        self._allCovariances =[]
        
    def _learnStep(self):
        """ one generation, also in the online case. """
        for k in range(self.batchsize):
            self.samples[k], self.fitnesses[k], self.densities[k] = self._produceNewSample()            
            if self.onlineLearning and self.numLearningSteps >= 1:
                self._updateWeightings()
                self._updateParameters(k)                
            if self._stoppingCriterion(): break
            
        if not self.onlineLearning:
            self._updateWeightings()
            self._updateParameters()                              
                    
        if self.storeAllCenters:
            self._allCenters.append(copy.deepcopy(self.mus))
        if self.storeAllDistributions:
            self._allCovariances.append(copy.deepcopy(self.sigmas))                    
                                                     
    def _produceNewSample(self):
        """ returns a new sample, its fitness and its densities """
        sample = self._generateSample()
        fit = self._oneEvaluation(sample)
        
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
        xdim = self.numParameters
        newMu = zeros(xdim)
        newSigma = zeros((xdim, xdim))
        for i in indices:
            newMu += weightings[i] * self.samples[i]
        # THIS LINE IS A HACK! REMOVE IT!
        newMu = self.forgetFactor * oldMu + (1-self.forgetFactor) * newMu
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
                    self.mus[c] = (1-lr) * self.mus[c] + lr * newMu
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
    