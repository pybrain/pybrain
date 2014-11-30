from __future__ import print_function

__author__ = 'Daan Wierstra and Tom Schaul'

from scipy import dot, rand, ones, eye, zeros, outer, isnan, multiply, argmax, product, log
from numpy.random import normal, multivariate_normal
from numpy import sort
from scipy.stats import norm
from copy import deepcopy

from pybrain.utilities import drawIndex, fListToString
from pybrain.tools.functions import multivariateNormalPdf
from pybrain.tools.rankingfunctions import TopLinearRanking
from pybrain.optimization.distributionbased.distributionbased import DistributionBasedOptimizer


class FEM(DistributionBasedOptimizer):
    """ Fitness Expectation-Maximization (PPSN 2008).
    """

    # fundamental parameters
    numberOfCenters = 1
    diagonalOnly = False
    forgetFactor = 0.1
    muMultiplier = 1.
    windowSize = 50

    adaptiveShaping = False

    shapingFunction = TopLinearRanking(topFraction=0.5)

    minimumCenterWeight = 0.01

    # advanced improvement parameters

    # elitism: always keep best mu in distribution
    elitism = False
    # sampleElitism: every $windowSize samples, produce best sample ever
    sampleElitism = False
    oneFifthRule = False

    useAnticipatedMeanShift = False

    # rank-mu update, presumably
    doMadnessUpdate = False

    mutative = False

    # initialization parameters
    rangemins = None
    rangemaxs = None
    initCovariances = None

    def _additionalInit(self):
        assert self.numberOfCenters == 1, 'Mixtures of Gaussians not supported yet.'

        xdim = self.numParameters
        self.alphas = ones(self.numberOfCenters) / float(self.numberOfCenters)
        self.mus = []
        self.sigmas = []

        if self.rangemins == None:
            self.rangemins = -ones(xdim)
        if self.rangemaxs == None:
            self.rangemaxs = ones(xdim)
        if self.initCovariances == None:
            if self.diagonalOnly:
                self.initCovariances = ones(xdim)
            else:
                self.initCovariances = eye(xdim)

        for _ in range(self.numberOfCenters):
            self.mus.append(rand(xdim) * (self.rangemaxs - self.rangemins) + self.rangemins)
            self.sigmas.append(dot(eye(xdim), self.initCovariances))

        self.samples = list(range(self.windowSize))
        self.fitnesses = zeros(self.windowSize)
        self.generation = 0
        self.allsamples = []
        self.muevals = []
        self.allmus = []
        self.allsigmas = []
        self.allalphas = []
        self.allUpdateSizes = []
        self.allfitnesses = []
        self.meanShifts = [zeros((self.numParameters)) for _ in range(self.numberOfCenters)]

        self._oneEvaluation(self._initEvaluable)


    def _produceNewSample(self):
        """ returns a new sample, its fitness and its densities """
        chosenOne = drawIndex(self.alphas, True)
        mu = self.mus[chosenOne]

        if self.useAnticipatedMeanShift:
            if len(self.allsamples) % 2 == 1 and len(self.allsamples) > 1:
                if not(self.elitism and chosenOne == self.bestChosenCenter):
                    mu += self.meanShifts[chosenOne]

        if self.diagonalOnly:
            sample = normal(mu, self.sigmas[chosenOne])
        else:
            sample = multivariate_normal(mu, self.sigmas[chosenOne])
        if self.sampleElitism and len(self.allsamples) > self.windowSize and len(self.allsamples) % self.windowSize == 0:
            sample = self.bestEvaluable.copy()
        fit = self._oneEvaluation(sample)

        if ((not self.minimize and fit >= self.bestEvaluation)
            or (self.minimize and fit <= self.bestEvaluation)
            or len(self.allsamples) == 0):
            # used to determine which center produced the current best
            self.bestChosenCenter = chosenOne
            self.bestSigma = self.sigmas[chosenOne].copy()
        if self.minimize:
            fit = -fit
        self.allfitnesses.append(fit)
        self.allsamples.append(sample)
        return sample, fit

    def _computeDensities(self, sample):
        """ compute densities, and normalize """
        densities = zeros(self.numberOfCenters)
        for c in range(self.numberOfCenters):
            if self.diagonalOnly:
                pdf = product([norm.pdf(x, self.mus[c][i], self.sigmas[c][i]) for i, x in enumerate(sample)])
            else:
                pdf = multivariateNormalPdf(sample, self.mus[c], self.sigmas[c])
            if pdf > 1e40:
                pdf = 1e40
            elif pdf < 1e-40:
                pdf = 1e-40
            if isnan(pdf):
                print('NaN!')
                pdf = 0.
            densities[c] = self.alphas[c] * pdf
        densities /= sum(densities)
        return densities

    def _computeUpdateSize(self, densities, sampleIndex):
        """ compute the  the center-update-size for each sample
        using transformed fitnesses """

        # determine (transformed) fitnesses
        transformedfitnesses = self.shapingFunction(self.fitnesses)
        # force renormaliziation
        transformedfitnesses /= max(transformedfitnesses)

        updateSize = transformedfitnesses[sampleIndex] * densities
        return updateSize * self.forgetFactor

    def _updateMus(self, updateSize, lastSample):
        for c in range(self.numberOfCenters):
            oldmu = self.mus[c]
            self.mus[c] *= 1. - self.muMultiplier * updateSize[c]
            self.mus[c] += self.muMultiplier * updateSize[c] * lastSample
            # don't update with the ones that were produced with a mean shift
            if ((self.useAnticipatedMeanShift and len(self.allsamples) % self.windowSize == 1)
                or (not self.useAnticipatedMeanShift and self.numberOfCenters > 1)):
                self.meanShifts[c] *= 1. - self.forgetFactor
                self.meanShifts[c] += self.mus[c] - oldmu

            if self.doMadnessUpdate and len(self.allsamples) > 2 * self.windowSize:
                self.mus[c] = zeros(self.numParameters)
                updateSum = 0.
                for i in range(self.windowSize):
                    self.mus[c] += self.allsamples[-i - 1] * self.allUpdateSizes[-i - 1][c]
                    updateSum += self.allUpdateSizes[-i - 1][c]
                self.mus[c] /= updateSum

        if self.elitism:
            # dirty hack! TODO: koshify
            self.mus[0] = self.bestEvaluable.copy()

    def _updateSigmas(self, updateSize, lastSample):
        for c in range(self.numberOfCenters):
            self.sigmas[c] *= (1. - updateSize[c])
            dif = self.mus[c] - lastSample
            if self.diagonalOnly:
                self.sigmas[c] += updateSize[c] * multiply(dif, dif)
            else:
                self.sigmas[c] += updateSize[c] * 1.2 * outer(dif, dif)

    def _updateAlphas(self, updateSize):
        for c in range(self.numberOfCenters):
            x = updateSize[c]
            x /= sum(updateSize)
            self.alphas[c] = (1.0 - self.forgetFactor) * self.alphas[c] + self.forgetFactor * x
        self.alphas /= sum(self.alphas)
        for c in range(self.numberOfCenters):
            if self.alphas[c] < self.minimumCenterWeight:
                # center-splitting
                if self.verbose:
                    print('Split!')
                bestCenter = argmax(self.alphas)
                totalWeight = self.alphas[c] + self.alphas[bestCenter]
                self.alphas[c] = totalWeight / 2
                self.alphas[bestCenter] = totalWeight / 2
                self.mus[c] = self.mus[bestCenter].copy()
                self.sigmas[c] = 4.0 * self.sigmas[bestCenter].copy()
                self.sigmas[bestCenter] *= 0.25
                break

    def _updateShaping(self):
        """ Daan: "This won't work. I like it!"  """
        assert self.numberOfCenters == 1
        possible = self.shapingFunction.getPossibleParameters(self.windowSize)
        matchValues = []
        pdfs = [multivariateNormalPdf(s, self.mus[0], self.sigmas[0])
                for s in self.samples]

        for p in possible:
            self.shapingFunction.setParameter(p)
            transformedFitnesses = self.shapingFunction(self.fitnesses)
            #transformedFitnesses /= sum(transformedFitnesses)
            sumValue = sum([x * log(y) for x, y in zip(pdfs, transformedFitnesses) if y > 0])
            normalization = sum([x * y for x, y in zip(pdfs, transformedFitnesses) if y > 0])
            matchValues.append(sumValue / normalization)


        self.shapingFunction.setParameter(possible[argmax(matchValues)])

        if len(self.allsamples) % 100 == 0:
            print((possible[argmax(matchValues)]))
            print((fListToString(matchValues, 3)))

    def _learnStep(self):
        k = len(self.allsamples) % self.windowSize
        sample, fit = self._produceNewSample()
        self.samples[k], self.fitnesses[k] = sample, fit
        self.generation += 1
        if len(self.allsamples) < self.windowSize:
            return
        if self.verbose and len(self.allsamples) % 100 == 0:
            print((len(self.allsamples), min(self.fitnesses), max(self.fitnesses)))
            # print(len(self.allsamples), min(self.fitnesses), max(self.fitnesses)#, self.alphas)

        updateSize = self._computeUpdateSize(self._computeDensities(sample), k)
        self.allUpdateSizes.append(deepcopy(updateSize))
        if sum(updateSize) > 0:
            # update parameters
            if self.numberOfCenters > 1:
                self._updateAlphas(updateSize)
            if not self.mutative:
                self._updateMus(updateSize, sample)
                self._updateSigmas(updateSize, sample)
            else:
                self._updateSigmas(updateSize, sample)
                self._updateMus(updateSize, sample)

        if self.adaptiveShaping:
            self._updateShaping()

        # storage, e.g. for plotting
        self.allalphas.append(deepcopy(self.alphas))
        self.allsigmas.append(deepcopy(self.sigmas))
        self.allmus.append(deepcopy(self.mus))

        if self.oneFifthRule and len(self.allsamples) % 10 == 0  and len(self.allsamples) > 2 * self.windowSize:
            lastBatch = self.allfitnesses[-self.windowSize:]
            secondLast = self.allfitnesses[-2 * self.windowSize:-self.windowSize]
            sortedLast = sort(lastBatch)
            sortedSecond = sort(secondLast)
            index = int(self.windowSize * 0.8)
            if sortedLast[index] >= sortedSecond[index]:
                self.sigmas = [1.2 * sigma for sigma in self.sigmas]
                #print("+")
            else:
                self.sigmas = [0.5 * sigma for sigma in self.sigmas]
                #print("-")

