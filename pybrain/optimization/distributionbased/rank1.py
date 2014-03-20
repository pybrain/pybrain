__author__ = 'Tom Schaul, Tobias Glasmachers'


from scipy import dot, array, randn,  exp, floor, log, sqrt, ones, multiply, log2

from pybrain.tools.rankingfunctions import HansenRanking
from pybrain.optimization.distributionbased.distributionbased import DistributionBasedOptimizer



class Rank1NES(DistributionBasedOptimizer):
    """ Natural Evolution Strategies with rank-1 covariance matrices. 
    
    See http://arxiv.org/abs/1106.1998 for a description. """
    
    # parameters, which can be set but have a good (adapted) default value
    centerLearningRate = 1.0
    scaleLearningRate = None
    covLearningRate = None 
    batchSize = None 
    uniformBaseline = True
    shapingFunction = HansenRanking()
    
    # fixed settings
    mustMaximize = True
    storeAllEvaluations = True    
    storeAllEvaluated = True    
    storeAllDistributions = True
    storeAllRates = True
    verboseGaps = 1
    initVariance = 1.
    varianceCutoff = 1e-20            

    
    def _additionalInit(self):
        # heuristic settings       
        if self.covLearningRate is None:
            self.covLearningRate = self._initLearningRate()
        if self.scaleLearningRate is None:
            self.scaleLearningRate = self.covLearningRate   
        if self.batchSize is None: 
            self.batchSize = self._initBatchSize()          
            
        # other initializations
        self._center = self._initEvaluable.copy()          
        self._logDetA = log(self.initVariance) / 2
        self._principalVector = randn(self.numParameters)
        self._principalVector /= sqrt(dot(self._principalVector, self._principalVector))
        self._allDistributions = [(self._center.copy(), self._principalVector.copy(), self._logDetA)]
        self.covLearningRate = 0.1
        self.batchSize = int(max(5,max(4*log2(self.numParameters),0.2*self.numParameters)))
        self.uniformBaseline = False
        self.scaleLearningRate = 0.1
    
    def _stoppingCriterion(self):
        if DistributionBasedOptimizer._stoppingCriterion(self):
            return True
        elif self._getMaxVariance < self.varianceCutoff:   
            return True
        else:
            return False
            
    @property
    def _getMaxVariance(self):
        return exp(self._logDetA * 2 / self.numParameters)
        
    def _initLearningRate(self):
        return 0.6 * (3 + log(self.numParameters)) / self.numParameters / sqrt(self.numParameters)
    
    def _initBatchSize(self):
        return 4 + int(floor(3 * log(self.numParameters)))               
            
    @property
    def _population(self):
        return [self._allEvaluated[i] for i in self._pointers]
        
    @property
    def _currentEvaluations(self):        
        fits = [self._allEvaluations[i] for i in self._pointers]
        if self._wasOpposed:
            fits = map(lambda x:-x, fits)
        return fits
                        
    def _produceSample(self):
        return randn(self.numParameters + 1)
    
    def _produceSamples(self):
        """ Append batch size new samples and evaluate them. """
        tmp = [self._sample2base(self._produceSample()) for _ in range(self.batchSize)]
        map(self._oneEvaluation, tmp)            
        self._pointers = list(range(len(self._allEvaluated) - self.batchSize, len(self._allEvaluated)))                    
        
    def _notify(self):
        """ Provide some feedback during the run. """
        if self.verbose:
            if self.numEvaluations % self.verboseGaps == 0:
                print('Step:', self.numLearningSteps, 'best:', self.bestEvaluation,
                    'logVar', round(self._logDetA, 3),
                    'log|vector|', round(log(dot(self._principalVector, self._principalVector))/2, 3))
                  
        if self.listener is not None:
            self.listener(self.bestEvaluable, self.bestEvaluation)
    
    def _learnStep(self):            
        # concatenations of y vector and z value
        samples = [self._produceSample() for _ in range(self.batchSize)]
                
        u = self._principalVector
        a = self._logDetA
        
        # unnamed in paper (y+zu), or x/exp(lambda)
        W = [s[:-1] + u * s[-1] for s in samples]   
        points = [self._center+exp(a) *w for w in W]
    
        map(self._oneEvaluation, points)  
                  
        self._pointers = list(range(len(self._allEvaluated) - self.batchSize, len(self._allEvaluated)))                            
        
        utilities = self.shapingFunction(self._currentEvaluations)
        utilities /= sum(utilities)  # make the utilities sum to 1
        if self.uniformBaseline:
            utilities -= 1. / self.batchSize    
        
        W = [w for i,w in enumerate(W) if utilities[i] != 0]
        utilities = [uw for uw in utilities if uw != 0]
                    
        dim = self.numParameters        
                 
        r = sqrt(dot(u, u))
        v = u / r
        c = log(r)        
        
        #inner products, but not scaled with exp(lambda) 
        wws = array([dot(w, w) for w in W])
        wvs = array([dot(v, w) for w in W])
        wv2s = array([wv ** 2 for wv in wvs])
        
        dCenter = exp(self._logDetA) * dot(utilities, W)
        self._center += self.centerLearningRate * dCenter       
        
        kp = ((r ** 2 - dim + 2) * wv2s - (r ** 2 + 1) * wws) / (2 * r * (dim - 1.))     

        # natural gradient on lambda, equation (5)
        da = 1. / (2 * (dim - 1)) * dot((wws - dim) - (wv2s - 1), utilities)
        
        # natural gradient on u, equation (6)
        du = dot(kp, utilities) * v + dot(multiply(wvs / r, utilities), W)
                
        # equation (7)
        dc = dot(du, v) / r
                
        # equation (8)
        dv = du / r - dc * v
        
        epsilon = min(self.covLearningRate, 2 * sqrt(r ** 2 / dot(du, du)))
        if dc > 0: 
            # additive update
            self._principalVector += epsilon * du
        else: 
            # multiplicative update
            # prevents instability            
            c += epsilon * dc
            v += epsilon * dv
            v /= sqrt(dot(v, v))
            r = exp(c)
            self._principalVector = r * v        
              
        self._lastLogDetA = self._logDetA
        self._logDetA += self.scaleLearningRate * da

        if self.storeAllDistributions:
            self._allDistributions.append((self._center.copy(), self._principalVector.copy(), self._logDetA))
    
    
def test():
    """ Rank-1 NEX easily solves high-dimensional Rosenbrock functions. """
    from pybrain.rl.environments.functions.unimodal import RosenbrockFunction
    dim = 40
    f = RosenbrockFunction(dim)
    x0 = -ones(dim)    
    l = Rank1NES(f, x0, verbose=True, verboseGaps=500)
    l.learn()
    
            
if __name__ == '__main__':
    test()