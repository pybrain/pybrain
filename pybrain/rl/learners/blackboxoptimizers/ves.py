__author__ = 'Daan Wierstra and Tom Schaul'

from pybrain.rl.learners.blackboxoptimizers.blackboxoptimizer import BlackBoxOptimizer
from pybrain.tools.rankingfunctions import TopLinearRanking
from pybrain.utilities import flat2triu, triu2flat

from scipy import eye, multiply, ones, dot, array, outer, rand, zeros, diag, reshape, randn, exp
from scipy.linalg import cholesky, inv, det


class VanillaGradientEvolutionStrategies(BlackBoxOptimizer):
    """ Vanilla gradient-based evolution strategy. """
    
    # mandatory parameters
    online = False
    learningRate = 1.
    learningRateSigma = None # default: the same than learningRate
    
    initialFactorSigma = None # default: identity matrix
    
    # NOT YET SUPPORTED:
    diagonalOnly = False
    
    batchSize = 100
    momentum = None
        
    elitism = False    
    
    shapingFunction = TopLinearRanking(topFraction = 0.5)

    # initialization parameters
    rangemins = None
    rangemaxs = None
    initCovariances = None
    
    vanillaScale = False
    
    # use of importance sampling to get away with fewer samples:
    importanceMixing = True
    forcedRefresh = 0.01
    
    def __init__(self, evaluator, evaluable, **parameters):
        BlackBoxOptimizer.__init__(self, evaluator, evaluable, **parameters)
        
        self.numParams = self.xdim + self.xdim * (self.xdim+1) / 2
                
        if self.momentum != None:
            self.momentumVector = zeros(self.numParams)
        if self.learningRateSigma == None:
            self.learningRateSigma = self.learningRate
        
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
        self.generation = 0
        self.evalsDone = 0
        
        # keeping track of history
        self.allSamples = []
        self.allFitnesses = []
        self.allPs = []
        
        self.allGenerated = [0]
        
        self.allCenters = [self.x.copy()]
        self.allFactorSigmas = [self.factorSigma.copy()]
        
        # for baseline computation
        self.phiSquareWindow = zeros((self.batchSize, self.numParams))
        

    def _produceNewSample(self, z = None, p = None):
        if z == None:
            p = randn(self.xdim)
            z = dot(self.factorSigma.T, p) + self.x
        if p == None:
            p = dot(inv(self.factorSigma).T, (z-self.x))            
        self.allPs.append(p)
        self.allSamples.append(z)
        fit = self.evaluator(z)
        self.evalsDone += 1
        self.allFitnesses.append(fit)
        if fit > self.bestEvaluation:
            self.bestEvaluation = fit
            self.bestEvaluable = z.copy()   
        return z, fit
    
    
    def _produceSamples(self):
        """ Append batchsize new samples and evaluate them. """
        if self.generation == 0 or not self.importanceMixing:
            for _ in range(self.batchSize):
                self._produceNewSample()
            self.allGenerated.append(self.batchSize + self.allGenerated[-1])
        else:
            olds = len(self.allSamples)
            oldDetFactorSigma = det(self.allFactorSigmas[-2])
            newDetFactorSigma = det(self.factorSigma)
            invA = inv(self.factorSigma)
    
            # All pdfs computed here are off by a coefficient of 1/power(2.0*pi, self.numParams/2.)
            # but as only their relative values matter, we ignore it.
            
            # stochastically reuse old samples, according to the change in distribution
            for s in range(olds-self.batchSize, olds):
                oldPdf = exp(-0.5*dot(self.allPs[s],self.allPs[s])) / oldDetFactorSigma
                sample = self.allSamples[s]
                newPs = dot(invA.T, (sample-self.x))
                newPdf = exp(-0.5*dot(newPs,newPs)) / newDetFactorSigma
                r = rand()
                if r < (1-self.forcedRefresh) * newPdf / oldPdf:
                    self.allSamples.append(sample)
                    self.allFitnesses.append(self.allFitnesses[s])
                    self.allPs.append(newPs)
                # never use only old samples
                if (olds+self.batchSize) - len(self.allSamples) < self.batchSize * self.forcedRefresh:
                    break
            self.allGenerated.append(self.batchSize - (len(self.allSamples) - olds) + self.allGenerated[-1])

            # add the remaining ones
            oldInvA = inv(self.allFactorSigmas[-2])
            while  len(self.allSamples) < olds + self.batchSize:
                r = rand()
                if r < self.forcedRefresh:
                    self._produceNewSample()
                else:
                    p = randn(self.xdim)
                    newPdf = exp(-0.5*dot(p,p)) / newDetFactorSigma
                    sample = dot(self.factorSigma.T, p) + self.x
                    oldPs = dot(oldInvA.T, (sample-self.allCenters[-2]))
                    oldPdf = exp(-0.5*dot(oldPs,oldPs)) / oldDetFactorSigma
                    if r < 1 - oldPdf/newPdf:
                        self._produceNewSample(sample, p)
                

    def _batchLearn(self, maxSteps):
        """ Batch learning. """
        while (self.evalsDone < maxSteps 
               and not self.bestEvaluation >= self.desiredEvaluation):
            # produce samples and evaluate them        
            try:
                self._produceSamples()
                
                # shape their fitnesses
                shapedFits = self.shapingFunction(self.allFitnesses[-self.batchSize:])
            
                # update parameters (unbiased: divide by batchsize)
                update = self._calcBatchUpdate(shapedFits) 
                if self.elitism:
                    self.x = self.bestEvaluable
                else:
                    self.x += self.learningRate * update[:self.xdim]
                self.factorSigma += self.learningRateSigma * flat2triu(update[self.xdim:], self.xdim)
                self.sigma = dot(self.factorSigma.T, self.factorSigma)
            
            except ValueError:
                print 'Numerical Instability. Stopping.'
                break
            
            if self._hasConverged():
                print 'Premature convergence. Stopping.'
                break
            
            if self.verbose:
                print 'G:', self.generation, 'Evals:', self.evalsDone, 'MaxG:', max(self.allFitnesses[-self.batchSize:])
                
            self.allCenters.append(self.x.copy())
            self.allFactorSigmas.append(self.factorSigma.copy())
            self.generation += 1


    def _learnStep(self):      
        """ Online learning. """    
        # produce one sample and evaluate        
        self._produceNewSample()
        if len(self.allSamples) <= self.batchSize:
            return
        
        # shape the fitnesses of the last samples
        shapedFits = self.shapingFunction(self.allFitnesses[-self.batchSize:])
        
        # update parameters
        update = self._calcOnlineUpdate(shapedFits)
        self.x += self.learningRate * update[:self.xdim]
        self.factorSigma += self.learningRateSigma * reshape(update[self.xdim:], (self.xdim, self.xdim))
        self.sigma = dot(self.factorSigma.T, self.factorSigma)
        
        if len(self.allSamples) % self.batchSize == 0:
            self.generation += 1
            print self.generation, len(self.allSamples), max(self.allFitnesses[-self.batchSize:])
            

    def _calcBatchUpdate(self, fitnesses):
        gradient = self._calcVanillaBatchGradient(self.allSamples[-self.batchSize:], fitnesses)
        if self.momentum != None:
            self.momentumVector *= self.momentum 
            self.momentumVector += gradient
            return self.momentumVector
        else:
            return gradient
            
    def _calcOnlineUpdate(self, fitnesses):
        gradient = self._calcVanillaOnlineGradient(self.allSamples[-1], fitnesses[-self.batchSize:])
        if self.momentum != None:
            self.momentumVector *= self.momentum 
            self.momentumVector += gradient
            return self.momentumVector
        else:
            return gradient

        
    def _logDerivX(self, sample, x, invSigma):
        return dot(invSigma, (sample - x))

    def _logDerivsX(self, samples, x, invSigma):
        samplesArray = array(samples)
        tmpX = multiply(x, ones((len(samplesArray), self.xdim)))
        return dot(invSigma, (samplesArray - tmpX).T).T
    
    
    def _logDerivFactorSigma(self, sample, x, invSigma, factorSigma):
        logDerivSigma = 0.5 * dot(dot(invSigma, outer(sample-x, sample-x)), invSigma) - 0.5 * invSigma
        if self.vanillaScale:
            logDerivSigma = multiply(outer(diag(abs(self.factorSigma)), diag(abs(self.factorSigma))), logDerivSigma)
        return triu2flat(dot(factorSigma, (logDerivSigma+logDerivSigma.T)))

        
    def _logDerivsFactorSigma(self, samples, x, invSigma, factorSigma):
        return [self._logDerivFactorSigma(sample, x, invSigma, factorSigma) for sample in samples]
                
                
    def _calcVanillaBatchGradient(self, samples, shapedfitnesses):
        invSigma = inv(self.sigma)
        
        phi = zeros((len(samples), self.numParams))
        phi[:, :self.xdim] = self._logDerivsX(samples, self.x, invSigma)
        logDerivFactorSigma = self._logDerivsFactorSigma(samples, self.x, invSigma, self.factorSigma)
        phi[:, self.xdim:] = array(logDerivFactorSigma)
        Rmat = outer(shapedfitnesses, ones(self.numParams))
        
        # optimal baseline
        self.phiSquareWindow = multiply(phi, phi)
        baselineMatrix = self._calcBaseline(shapedfitnesses)
        
        gradient = sum(multiply(phi, (Rmat - baselineMatrix)), 0)
        return gradient    
        
    def _calcVanillaOnlineGradient(self, sample, shapedfitnesses):
        invSigma = inv(self.sigma)
        phi = zeros(self.numParams)
        phi[:self.xdim] = self._logDerivX(sample, self.x, invSigma)    
        logDerivSigma = self._logDerivFactorSigma(sample, self.x, invSigma, self.factorSigma)
        phi[self.xdim:] = logDerivSigma.flatten()
        index = len(self.allSamples) % self.batchSize
        self.phiSquareWindow[index] = multiply(phi, phi)
        baseline = self._calcBaseline(shapedfitnesses)
        gradient = multiply((ones(self.numParams)*shapedfitnesses[-1] - baseline), phi)
        return gradient
    
    def _calcBaseline(self, shapedfitnesses):
        paramWeightings = dot(ones(self.batchSize), self.phiSquareWindow)
        baseline = dot(shapedfitnesses, self.phiSquareWindow) / paramWeightings
        return baseline
        
    def _hasConverged(self):
        """ When the largest eigenvalue is smaller than 10e-20, we assume the 
        algorithms has converged. """
        eigs = abs(diag(self.factorSigma))
        return min(eigs) < 1e-10
        
    def _revertToSafety(self):
        """ When encountering a bad matrix, this is how we revert to a safe one. """
        self.factorSigma = eye(self.xdim)
        self.x = self.bestEvaluable
        self.allFactorSigmas[-1][:] = self.factorSigma
        self.sigma = dot(self.factorSigma.T, self.factorSigma)
