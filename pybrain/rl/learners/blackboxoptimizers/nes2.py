__author__ = 'Daan Wierstra and Tom Schaul'


from scipy import eye, multiply, ones, dot, array, outer, rand, ravel, zeros, diag, tril, triu, reshape, average
from scipy.linalg import cholesky,  pinv2, inv
from numpy.random import multivariate_normal

from blackboxoptimizer import BlackBoxOptimizer
from pybrain.tools.rankingfunctions import TopLinearRanking

# TODO:
# batch vanilla working
# batch natural grads working
# online vanilla
# online/rls natural
#
#
class NaturalEvolutionStrategies2(BlackBoxOptimizer):
    """ do the optimization using natural fitness gradients.
    New, cleaner version. With more options too! But no multiple center support anymore.
    Unfinished...
    """
    
    # mandatory parameters
    online = False
    learningRate = 0.05
    
    initialFactorSigma = None
    diagonalOnly = False
    
    batchSize = 100
    momentum = None
    
    windowSize = None # default: batch size
    
    shapingFunction = TopLinearRanking(topFraction = 0.8)

    # initalization parameters
    rangemins = None
    rangemaxs = None
    initCovariances = None
    
    def __init__(self, evaluator, evaluable, **parameters):
        BlackBoxOptimizer.__init__(self, evaluator, evaluable, **parameters)
        # TODO specify or not
        #self.x = self.x0.copy()  
        
        self.numParams = self.xdim * (self.xdim+1)
        
        if self.momentum != None:
            self.momentumVector = zeros(self.numParams)
        
        if self.rangemins == None:
            self.rangemins = -ones(self.xdim)
        if self.rangemaxs == None:
            self.rangemaxs = ones(self.xdim)
        if self.initCovariances == None:
            if self.diagonalOnly:
                self.initCovariances = ones(self.xdim)
            else:
                self.initCovariances = eye(self.xdim)

        self.x = rand(self.xdim) * (self.rangemaxs-self.rangemins) + self.rangemins
        self.sigma = dot(eye(self.xdim), self.initCovariances)
        self.factorSigma = cholesky(self.sigma)
            
        self.reset()


    def reset(self):
        # keeping track of history
        self.allSamples = []
        self.allFitnesses = []

        # for baseline computation
        self.phiSquareWindow = zeros((self.batchSize, self.numParams))
        self.generation = 0
        self.allSamples = []


    def _produceNewSample(self):
        z = multivariate_normal(self.x, self.sigma)
        self.allSamples.append(z)
        fit = self.evaluator(z)
        self.allFitnesses.append(fit)
        if fit > self.bestEvaluation:
            self.bestEvaluation = fit
            self.bestEvaluable = z.copy()   
        #print len(self.allFitnesses), ':', res, z 
        return z, fit


    def _batchLearn(self, maxSteps):
        while len(self.allSamples) < maxSteps:            
            for dummy in range(self.batchSize):
                self._produceNewSample()
            shapedFits = self.shapingFunction(self.allFitnesses[-self.batchSize:])
            gradient = self._calcVanillaGradient(self.allSamples[-self.batchSize:], shapedFits)
            if self.momentum != None:
                self.momentumVector *= self.momentum 
                self.momentumVector += gradient
                gradient = self.momentumVector
            
            self.x += self.learningRate * gradient[:self.xdim]
            self.factorSigma += self.learningRate * reshape(gradient[self.xdim:], (self.xdim, self.xdim))
            #print 'fsigma', self.factorSigma
            self.sigma = dot(self.factorSigma.T, self.factorSigma)
            print 'sigma', self.sigma
            
            print self.generation, len(self.allSamples), max(self.allFitnesses[-self.batchSize:])
            self.generation += 1


    def _learnStep(self):
        pass



    def _logDerivX(self, samples, x, invSigma):
        samplesArray = array(samples)
        tmpX = multiply(x, ones((len(samplesArray), self.xdim)))
        return dot(invSigma, (samplesArray - tmpX).T).T
    
    
    def _logDerivFactorSigma(self, samples, x, invSigma, factorSigma):
        res = []
        for sample in samples:
            logDerivSigma = 0.5 * dot(dot(invSigma, outer(sample-x, sample-x)), invSigma) - 0.5 * invSigma
            #logDerivSigma = multiply(outer(diag(abs(self.factorSigma)), diag(abs(self.factorSigma))), logDerivSigma)
            #logDerivSigma = multiply(outer(diag(abs(self.factorSigma)), diag(abs(self.factorSigma))), logDerivSigma)
            #logDerivSigma = multiply(abs(self.sigma), logDerivSigma)
            logDerivFactorSigma = dot(factorSigma, (logDerivSigma+logDerivSigma.T))
            #logDerivFactorSigma = multiply(outer(diag(self.factorSigma),diag(self.factorSigma)), logDerivFactorSigma)
            res.append(logDerivFactorSigma)
        return res
                
                
    def _calcVanillaGradient(self, samples, shapedfitnesses):
        phi = zeros((len(samples), self.numParams))
        invSigma = inv(self.sigma)
        phi[:, :self.xdim] = self._logDerivX(samples, self.x, invSigma)
        logDerivSigma = self._logDerivFactorSigma(samples, self.x, invSigma, self.factorSigma)
        # TODO: potentially remove half of the parameters?
        phi[:, self.xdim:] = array(map(ravel, logDerivSigma))
        
        Rmat = outer(shapedfitnesses, ones(self.numParams))
        baselineMatrix = ones((len(samples), self.numParams)) * average(shapedfitnesses)
        gradient = dot(ones(len(samples)), multiply(phi, (Rmat - baselineMatrix)))
        return gradient        
    
    
    #def _updateBaseline(self, phi):
    #    phiSquare = multiply(phi, phi)
    #    fitnesses = self.allFitnesses[-self.batchSize:]
    #   
    #    if self.online:
    #        index = len(self.allSamples) % self.batchSize
    #       self.phiSquareWindow[index:index+len(phi)] = phiSquare
    #       fits = fitnesses[index:][:]
    #        fits.extend(fitnesses[:index])
    #        fits = self.shapingFunction(fits)
    #        
    #        phiSquare = self.phiSquareWindow
    #    else:
    #        fits = self.shapingFunction(fitnesses)
    #        
    #    paramWeightings = dot(ones(self.batchSize), phiSquare)
    #    baseline = dot(fits, phiSquare) / paramWeightings
    #    
    #    #print 'w', paramWeightings
    #    #print 'b', baseline
    #    assert min(paramWeightings) > 0    
    #    return  baseline
        
