__author__ = 'Daan Wierstra and Tom Schaul'

from scipy import eye, multiply, ones, dot, array, outer, rand, zeros, diag, randn, exp
from scipy.linalg import cholesky, inv, det

from pybrain.optimization.distributionbased.distributionbased import DistributionBasedOptimizer
from pybrain.tools.rankingfunctions import TopLinearRanking
from pybrain.utilities import flat2triu, triu2flat
from pybrain.auxiliary import importanceMixing


class VanillaGradientEvolutionStrategies(DistributionBasedOptimizer):
    """ Vanilla gradient-based evolution strategy. """

    # mandatory parameters
    online = False
    learningRate = 0.01
    learningRateSigma = None # default: the same than learningRate

    initialFactorSigma = None # default: identity matrix

    # NOT YET SUPPORTED:
    diagonalOnly = False

    batchSize = 100
    momentum = None

    elitism = False

    shapingFunction = TopLinearRanking(topFraction=0.5)

    # initialization parameters
    rangemins = None
    rangemaxs = None
    initCovariances = None

    vanillaScale = False

    # use of importance sampling to get away with fewer samples:
    importanceMixing = True
    forcedRefresh = 0.01

    mustMaximize = True

    def _additionalInit(self):
        xdim = self.numParameters
        assert not self.diagonalOnly, 'Diagonal-only not yet supported'
        self.numDistrParams = xdim + xdim * (xdim + 1) / 2

        if self.momentum != None:
            self.momentumVector = zeros(self.numDistrParams)
        if self.learningRateSigma == None:
            self.learningRateSigma = self.learningRate
        if self.batchSize is None:
            self.batchSize = 10 * self.numParameters

        if self.rangemins == None:
            self.rangemins = -ones(xdim)
        if self.rangemaxs == None:
            self.rangemaxs = ones(xdim)
        if self.initCovariances == None:
            if self.diagonalOnly:
                self.initCovariances = ones(xdim)
            else:
                self.initCovariances = eye(xdim)

        self.x = rand(xdim) * (self.rangemaxs - self.rangemins) + self.rangemins
        self.sigma = dot(eye(xdim), self.initCovariances)
        self.factorSigma = cholesky(self.sigma)

        # keeping track of history
        self.allSamples = []
        self.allFitnesses = []

        self.allGenerated = [0]

        self.allCenters = [self.x.copy()]
        self.allFactorSigmas = [self.factorSigma.copy()]

        # for baseline computation
        self.phiSquareWindow = zeros((self.batchSize, self.numDistrParams))

        if self.storeAllDistributions:
            self._allDistributions = [(self.x.copy(), self.sigma.copy())]

    def _produceNewSample(self, z=None):
        if z == None:
            p = randn(self.numParameters)
            z = dot(self.factorSigma.T, p) + self.x
        self.allSamples.append(z)
        fit = self._oneEvaluation(z)
        self.allFitnesses.append(fit)
        return z, fit

    def _produceSamples(self):
        """ Append batchsize new samples and evaluate them. """
        if self.numLearningSteps == 0 or not self.importanceMixing:
            for _ in range(self.batchSize):
                self._produceNewSample()
            self.allGenerated.append(self.batchSize + self.allGenerated[-1])

        # using new importance mixing code
        else:
            oldpoints = self.allSamples[-self.batchSize:]
            oldDetFactorSigma = det(self.allFactorSigmas[-2])
            newDetFactorSigma = det(self.factorSigma)
            invA = inv(self.factorSigma)
            offset = len(self.allSamples) - self.batchSize
            oldInvA = inv(self.allFactorSigmas[-2])
            oldX = self.allCenters[-2]

            def oldpdf(s):
                p = dot(oldInvA.T, (s- oldX))
                return exp(-0.5 * dot(p, p)) / oldDetFactorSigma

            def newpdf(s):
                p = dot(invA.T, (s - self.x))
                return exp(-0.5 * dot(p, p)) / newDetFactorSigma

            def newSample():
                p = randn(self.numParameters)
                return dot(self.factorSigma.T, p) + self.x

            reused, newpoints = importanceMixing(oldpoints, oldpdf, newpdf,
                                                 newSample, self.forcedRefresh)

            self.allGenerated.append(self.allGenerated[-1]+len(newpoints))

            for i in reused:
                self.allSamples.append(self.allSamples[offset+i])
                self.allFitnesses.append(self.allFitnesses[offset+i])
            for s in newpoints:
                self._produceNewSample(s)

    def _learnStep(self):
        if self.online:
            self._onlineLearn()
        else:
            self._batchLearn()

    def _batchLearn(self):
        """ Batch learning. """
        xdim = self.numParameters
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
                self.x += self.learningRate * update[:xdim]
            self.factorSigma += self.learningRateSigma * flat2triu(update[xdim:], xdim)
            self.sigma = dot(self.factorSigma.T, self.factorSigma)

        except ValueError:
            print('Numerical Instability. Stopping.')
            self.maxLearningSteps = self.numLearningSteps

        if self._hasConverged():
            print('Premature convergence. Stopping.')
            self.maxLearningSteps = self.numLearningSteps

        if self.verbose:
            print('Evals:', self.numEvaluations,)

        self.allCenters.append(self.x.copy())
        self.allFactorSigmas.append(self.factorSigma.copy())

        if self.storeAllDistributions:
            self._allDistributions.append((self.x.copy(), self.sigma.copy()))


    def _onlineLearn(self):
        """ Online learning. """
        # produce one sample and evaluate
        xdim = self.numParameters
        self._produceNewSample()
        if len(self.allSamples) <= self.batchSize:
            return

        # shape the fitnesses of the last samples
        shapedFits = self.shapingFunction(self.allFitnesses[-self.batchSize:])

        # update parameters
        update = self._calcOnlineUpdate(shapedFits)
        self.x += self.learningRate * update[:xdim]
        self.factorSigma += self.learningRateSigma * flat2triu(update[xdim:], xdim)
        self.sigma = dot(self.factorSigma.T, self.factorSigma)

        if self.storeAllDistributions:
            self._allDistributions.append(self.x.copy(), self.sigma.copy())

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
        tmpX = multiply(x, ones((len(samplesArray), self.numParameters)))
        return dot(invSigma, (samplesArray - tmpX).T).T

    def _logDerivFactorSigma(self, sample, x, invSigma, factorSigma):
        logDerivSigma = 0.5 * dot(dot(invSigma, outer(sample - x, sample - x)), invSigma) - 0.5 * invSigma
        if self.vanillaScale:
            logDerivSigma = multiply(outer(diag(abs(self.factorSigma)), diag(abs(self.factorSigma))), logDerivSigma)
        return triu2flat(dot(factorSigma, (logDerivSigma + logDerivSigma.T)))

    def _logDerivsFactorSigma(self, samples, x, invSigma, factorSigma):
        return [self._logDerivFactorSigma(sample, x, invSigma, factorSigma) for sample in samples]

    def _calcVanillaBatchGradient(self, samples, shapedfitnesses):
        invSigma = inv(self.sigma)

        phi = zeros((len(samples), self.numDistrParams))
        phi[:, :self.numParameters] = self._logDerivsX(samples, self.x, invSigma)
        logDerivFactorSigma = self._logDerivsFactorSigma(samples, self.x, invSigma, self.factorSigma)
        phi[:, self.numParameters:] = array(logDerivFactorSigma)
        Rmat = outer(shapedfitnesses, ones(self.numDistrParams))

        # optimal baseline
        self.phiSquareWindow = multiply(phi, phi)
        baselineMatrix = self._calcBaseline(shapedfitnesses)

        gradient = sum(multiply(phi, (Rmat - baselineMatrix)), 0)
        return gradient

    def _calcVanillaOnlineGradient(self, sample, shapedfitnesses):
        invSigma = inv(self.sigma)
        phi = zeros(self.numDistrParams)
        phi[:self.numParameters] = self._logDerivX(sample, self.x, invSigma)
        logDerivSigma = self._logDerivFactorSigma(sample, self.x, invSigma, self.factorSigma)
        phi[self.numParameters:] = logDerivSigma.flatten()
        index = len(self.allSamples) % self.batchSize
        self.phiSquareWindow[index] = multiply(phi, phi)
        baseline = self._calcBaseline(shapedfitnesses)
        gradient = multiply((ones(self.numDistrParams) * shapedfitnesses[-1] - baseline), phi)
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
        self.factorSigma = eye(self.numParameters)
        self.x = self.bestEvaluable
        self.allFactorSigmas[-1][:] = self.factorSigma
        self.sigma = dot(self.factorSigma.T, self.factorSigma)

