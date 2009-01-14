__author__ = 'Daan Wierstra and Tom Schaul'


from scipy import eye, multiply, ones, dot, array, outer, rand, ravel, zeros, diag, tril, triu, reshape, average, maximum, sqrt
from scipy.linalg import cholesky,  pinv2, inv
from numpy.random import multivariate_normal
from numpy import sign

from blackboxoptimizer import BlackBoxOptimizer
from pybrain.tools.rankingfunctions import TopLinearRanking, SmoothGiniRanking


#
#
class QuasiNewtonEvolutionStrategies(BlackBoxOptimizer):
    """ do the optimization using quasi newton (schraudolph's) fitness gradients.
    New, cleaner version. With more options too! But no multiple center support anymore.
    Unfinished...
    """
    
    # mandatory parameters
    online = True
    
    initialFactorSigma = None
    diagonalOnly = False
    
    batchSize = 100
    windowSize = 100
    
    #shapingFunction = SmoothGiniRanking()#TopLinearRanking(topFraction = 0.2)
    shapingFunction = TopLinearRanking(topFraction = 0.2)

    # initalization parameters
    rangemins = None
    rangemaxs = None
    initCovariances = None
    
    def __init__(self, evaluator, evaluable, **parameters):
        BlackBoxOptimizer.__init__(self, evaluator, evaluable, **parameters)
        # TODO specify or not
        #self.x = self.x0.copy()  
        
        self.numParams = self.xdim * (self.xdim+1)
        
        self.c = 0.1
        self.l = 1e-20
        self.tau = 10.0
        self.epsilon = 1e-10
        self.eta = 0.8#10.1
        self.starteta = self.eta
        
        self.B = self.epsilon * eye(self.numParams, self.numParams)
                    
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
        pass

    def _learnStep(self):          
        self._produceNewSample()
        if len(self.allSamples) < self.windowSize:
            return
        if (len(self.allSamples) - self.windowSize) % self.batchSize != 0:
            return
        shapedFits = self.shapingFunction(self.allFitnesses[-self.batchSize:])
        gradient = -self._calcGradient(self.x, self.factorSigma)
        #print "gradient", gradient
        self.p = -dot(self.B, gradient)
        #print self.B
        #print "p", self.p
        self.eta = self.starteta# * self.tau/(self.tau + self.generation)
        self.s = (self.eta/self.c) * self.p
        self.x += self.s[:self.xdim]
        self.factorSigma += 0.0 * reshape(self.s[self.xdim:], (self.xdim, self.xdim))
        self.sigma = dot(self.factorSigma.T, self.factorSigma)
        newgradient = -self._calcGradient(self.x, self.factorSigma)
        self.y = newgradient - gradient + self.l * self.s
        # TODO: execute this only the first-time -- check!!
        if len(self.allSamples) <= self.windowSize+1:
            self.B = (dot(self.s,self.y)/dot(self.y,self.y)) * eye(self.numParams)
            print "B", len(self.allSamples)
        self.rho = abs(1.0/(dot(self.s,self.y)))
        h1 = (eye(self.numParams) - self.rho * outer(self.s,self.y))
        h2 = (eye(self.numParams) - self.rho * outer(self.y,self.s))
        self.B = dot(dot(h1, self.B), h2) + self.c * self.rho * outer(self.s,self.s)
        #print 'ONLINE sigma', self.sigma
        if len(self.allSamples) % self.batchSize == 0:
            self.generation += 1
            print self.generation, len(self.allSamples), max(self.allFitnesses[-self.batchSize:])
            #print self.sigma

    
        
        
    def _logDerivX(self, sample, x, invSigma):
        return dot(invSigma, (sample - x))
    
    
    def _logDerivFactorSigma(self, sample, x, invSigma, factorSigma):
        logDerivSigma = 0.5 * dot(dot(invSigma, outer(sample-x, sample-x)), invSigma) - 0.5 * invSigma
        return dot(factorSigma, (logDerivSigma+logDerivSigma.T))

                        
                
        
    def _calcGradient(self, x, factorSigma):
        #print dot(factorSigma.T, factorSigma)
        invSigma = inv(dot(factorSigma.T, factorSigma))
        shapedfitnesses = self.shapingFunction(self.allFitnesses[-self.windowSize:])
        gradient = zeros(self.numParams)
        baseline = average(shapedfitnesses)
        for i in range(self.batchSize):
            phi = zeros(self.numParams)
            phi[:self.xdim] = self._logDerivX(self.allSamples[-i-1], x, invSigma)    
            logDerivSigma = self._logDerivFactorSigma(self.allSamples[-i-1], x, invSigma, factorSigma)
            phi[self.xdim:] = logDerivSigma.flatten()
            gradient += (shapedfitnesses[-i-1] - baseline) * phi
        return gradient


