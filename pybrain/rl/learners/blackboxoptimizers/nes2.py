__author__ = 'Daan Wierstra and Tom Schaul'


from scipy import eye, multiply, ones, dot, array, outer, rand, ravel, zeros, diag, tril, triu, reshape
from scipy.linalg import cholesky,  pinv2, inv
from numpy.random import multivariate_normal

from blackboxoptimizer import BlackBoxOptimizer
from pybrain.tools.rankingfunctions import TopLinearRanking


class NaturalEvolutionStrategies2(BlackBoxOptimizer):
    """ do the optimization using natural fitness gradients.
    New, cleaner version. With more options too! But no multiple center support anymore.
    """
    
    # mandatory parameters
    online = False
    learningRate = 0.05
    
    initialFactorSigma = None
    onlyDiagonal = False
    
    batchSize = 100
    momentum = None
    
    windowSize = None # default: batch size
    
    fitnessShaping = TopLinearRanking()
    
    def __init__(self, evaluator, evaluable, **parameters):
        BlackBoxOptimizer.__init__(self, evaluator, evaluable, **parameters)
        self.x = self.x0.copy()
        if self.initialFactorSigma == None:
            self.factorSigma = eye(self.xdim)
        else:
            self.factorSigma = self.initialFactorSigma.copy()
        self.sigma = dot(self.factorSigma, self.factorSigma)        
        
        self.numParams = self.xdim * (self.xdim+1)
        
        # keeping track of history
        self.allSamples = []
        self.allFitnesses = []
        
        # for baseline computation
        self.phiSquareWindow = zeros((self.batchSize, self.numParams))
        
        if self.momentum != None:
            self.momentumVector = zeros(self.numParams)
            
    def _oneSample(self):
        z = multivariate_normal(self.x, self.sigma)
        self.allSamples.append(z)
        res = self.evaluator(z)
        self.allFitnesses.append(res)
        if res > self.bestEvaluation:
            self.bestEvaluation = res
            self.bestEvaluable = z   
        print len(self.allFitnesses), ':', res, z 
    
    def _batchLearn(self, maxSteps):
        while len(self.allSamples) < maxSteps:
            for dummy in range(self.batchSize):
                self._oneSample()
            gradient = self._calcVanillaGradient(self.allSamples[-self.batchSize:], 
                                                 self.allFitnesses[-self.batchSize:])
            if self.momentum != None:
                self.momentumVector *= self.momentum 
                self.momentumVector += gradient 
                gradient = self.momentumVector
                
            self.x += self.learningRate * gradient[:self.xdim]
            self.factorSigma += self.learningRate * reshape(gradient[self.xdim:], (self.xdim, self.xdim))
            print 'fsigma', self.factorSigma
            self.sigma = dot(self.factorSigma, self.factorSigma)
            print 'sigma', self.sigma
        
    def _learnStep(self):
        
        
        
        #self.bestEvaluable, self.bestEvaluation
        pass
        
    
    
    def _logDerivX(self, samples, x, invSigma):
        samplesArray = array(samples)
        tmpX = multiply(x, ones((len(samplesArray), self.xdim)))
        return dot(invSigma, (samplesArray - tmpX).T).T
    
    def _logDerivFactorSigma(self, samples, x, invSigma, factorSigma):
        res = []
        for sample in samples:
            logDerivSigma = 0.5 * dot(dot(invSigma, outer(sample-x, sample-x)), invSigma) - 0.5 * invSigma
            #logDerivSigma = multiply(abs(self.sigma), logDerivSigma)
            logDerivFactorSigma = dot(factorSigma, (logDerivSigma+logDerivSigma.T))
            res.append(logDerivFactorSigma)
        return res
                
    def _calcVanillaGradient(self, samples, fitnesses):
        phi = zeros((len(samples), self.numParams))
        invSigma = inv(dot(self.factorSigma, self.factorSigma))
        phi[:, :self.xdim] = self._logDerivX(samples, self.x, invSigma)
        logDerivSigma = self._logDerivFactorSigma(samples, self.x, invSigma, self.factorSigma)
        # TODO: potentially remove half of the parameters?
        phi[:, self.xdim:] = array(map(ravel, logDerivSigma))
        
        Rmat = outer(fitnesses, ones(self.numParams))
        Bmat = outer(ones(len(samples)), self._updateBaseline(phi))
        gradient = dot(ones(len(samples)), phi*(Rmat-Bmat))
        return gradient        
    
    def _updateBaseline(self, phi):
        phiSquare = multiply(phi, phi)
        fitnesses = self.allFitnesses[-self.batchSize:]
        
        if self.online:
            index = len(self.allSamples) % self.batchSize
            self.phiSquareWindow[index:index+len(phi)] = phiSquare
            fits = fitnesses[index:][:]
            fits.extend(fitnesses[:index])
            phiSquare = self.phiSquareWindow
        else:
            fits = fitnesses            
            
        paramWeightings = dot(ones(self.batchSize), phiSquare)            
        baseline = dot(fits, phiSquare) / paramWeightings
        
        print 'w', paramWeightings
        print 'b', baseline
        assert min(paramWeightings) > 0    
        return  baseline
        



















if False:
    dim = 3
        
    def inTri(A):
        res = []
        for i in range(dim):
            for j in range(i+1):
                res.append(A[i,j])
        return array(res)
    
    def deTri(A):
        res = zeros((dim, dim))
        finger = 0
        for i in range(dim):
            for j in range(i+1):
                res[i,j] = A[finger]
                finger += 1
        return res
    
    def magic(A):
        res = inTri(2 * A - diag(diag(A)))
        return res
    
    
        
    ss = 1
    tmp = rand(dim,dim)
    Z = dot(tmp.T, tmp)
    A = cholesky(Z)
    R = rand(ss)
    logDS = rand(dim,dim)
    logDA = dot(A, (logDS+logDS.T))
    #print 'Z', Z
    #print 'R', R
    #print 'logDA', logDA
    
    Phi1 = array([ravel(logDA)]*ss)
    d1 = dot(pinv2(Phi1), R).reshape(dim, dim)
    messA = A + d1
    Z1 = dot(messA.T, messA)
    A1 = cholesky(Z1)
    
    Phi2 = array([ravel(magic(logDA))]*ss)
    d2 = dot(pinv2(Phi2), R)
    d2 = deTri(d2)
    A2 = A + d2
    
    d3 = tril(d1)+triu(d1).T-diag(diag(d1))
    A3 = A + d3
    
    
    print 'Phi1.shape', Phi1.shape
    print 'Phi2.shape', Phi2.shape
    print
    print 'Phi1', Phi1
    print 'Phi2', Phi2
    print
    print 'd1', d1
    print 'd2', d2
    print 'd3', d3
    print
    print 'A', A
    print 'messA',  messA
    print 
    print 'A1', A1
    print 'A2', A2
    print 'A3', A3


