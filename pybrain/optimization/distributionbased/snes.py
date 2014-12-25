from __future__ import print_function

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.optimization.distributionbased.distributionbased import DistributionBasedOptimizer
from scipy import dot, exp, log, sqrt, floor, ones, randn
from pybrain.tools.rankingfunctions import HansenRanking


class SNES(DistributionBasedOptimizer):
    """ Separable NES (diagonal). 
    [As described in Schaul, Glasmachers and Schmidhuber (GECCO'11)]
    """
    
    # parameters, which can be set but have a good (adapted) default value
    centerLearningRate = 1.0
    covLearningRate = None     
    batchSize = None     
    uniformBaseline = True
    shapingFunction = HansenRanking()
    initVariance = 1.
    
    # fixed settings
    mustMaximize = True
    storeAllEvaluations = True    
    storeAllEvaluated = True
    
    # for very long runs, we don't want to run out of memory
    clearStorage = False    
            
    # minimal setting where to abort the search
    varianceCutoff = 1e-20
            
    def _stoppingCriterion(self):
        if DistributionBasedOptimizer._stoppingCriterion(self):
            return True
        elif max(abs(self._sigmas)) < self.varianceCutoff:   
            return True
        else:
            return False
            
    def _initLearningRate(self):
        """ Careful, robust default value. """
        return 0.6 * (3 + log(self.numParameters)) / 3 / sqrt(self.numParameters)
        
    def _initBatchSize(self):
        """ as in CMA-ES """
        return 4 + int(floor(3 * log(self.numParameters)))    
    
    def _additionalInit(self):
        if self.covLearningRate is None:
            self.covLearningRate = self._initLearningRate()        
        if self.batchSize is None:
            self.batchSize = self._initBatchSize()                           
            
        self._center = self._initEvaluable.copy()          
        self._sigmas = ones(self.numParameters) * self.initVariance
    
    @property
    def _population(self):
        if self._wasUnwrapped:
            return [self._allEvaluated[i].params for i in self._pointers]
        else:
            return [self._allEvaluated[i] for i in self._pointers]
            
    @property
    def _currentEvaluations(self):        
        fits = [self._allEvaluations[i] for i in self._pointers]
        if self._wasOpposed:
            fits = [-x for x in fits]
        return fits
                        
    def _produceSample(self):
        return randn(self.numParameters)
            
    def _sample2base(self, sample):       
        """ How does a sample look in the outside (base problem) coordinate system? """ 
        return self._sigmas * sample + self._center
              
    def _base2sample(self, e):
        """ How does the point look in the present one reference coordinates? """
        return (e - self._center) / self._sigmas
    
    def _produceSamples(self):
        """ Append batch size new samples and evaluate them. """
        if self.clearStorage:
            self._allEvaluated = []
            self._allEvaluations = []
            
        tmp = [self._sample2base(self._produceSample()) for _ in range(self.batchSize)]
        list(map(self._oneEvaluation, tmp))            
        self._pointers = list(range(len(self._allEvaluated) - self.batchSize, len(self._allEvaluated)))                    
            
    def _learnStep(self):
        # produce samples
        self._produceSamples()
        samples = list(map(self._base2sample, self._population)) 
        
        #compute utilities
        utilities = self.shapingFunction(self._currentEvaluations)
        utilities /= sum(utilities)  # make the utilities sum to 1
        if self.uniformBaseline:
            utilities -= 1. / self.batchSize                           
                    
        # update center
        dCenter = dot(utilities, samples)
        self._center += self.centerLearningRate * self._sigmas * dCenter
        
        # update variances
        covGradient = dot(utilities, [s ** 2 - 1 for s in samples])        
        dA = 0.5 * self.covLearningRate * covGradient                                
        self._sigmas = self._sigmas * exp(dA)            
        
        
if __name__ == "__main__":
    from pybrain.rl.environments.functions.unimodal import ElliFunction
    print((SNES(ElliFunction(100), ones(100), verbose=True).learn()))
    