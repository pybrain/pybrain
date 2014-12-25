from __future__ import print_function

__author__ = 'Tom Schaul, tom@idsia.ch; Sun Yi, yi@idsia.ch'

from numpy import floor, log, eye, zeros, array, sqrt, sum, dot, tile, outer, real
from numpy import exp, diag, power, ravel
from numpy.linalg import eig, norm
from numpy.random import randn

from pybrain.optimization.optimizer import ContinuousOptimizer


class CMAES(ContinuousOptimizer):
    """ CMA-ES: Evolution Strategy with Covariance Matrix Adaptation for
    nonlinear function minimization.
    This code is a close transcription of the provided matlab code.
    """

    mustMinimize = True
    stopPrecision = 1e-6

    storeAllCenters = False
    initStepSize = 0.5

    def _additionalInit(self):
        self.center = self._initEvaluable
        self.stepSize = self.initStepSize  # coordinate wise standard deviation (sigma)
        if self.storeAllCenters:
            self._allCenters = []

        # Strategy parameter setting: Selection
        # population size, offspring number
        self.mu = int(floor(self.batchSize / 2))        # number of parents/points for recombination
        self.weights = log(self.mu + 1) - log(array(range(1, self.mu + 1)))      # use array
        self.weights /= sum(self.weights)     # normalize recombination weights array
        self.muEff = sum(self.weights) ** 2 / sum(power(self.weights, 2)) # variance-effective size of mu

        # Strategy parameter setting: Adaptation
        self.cumCov = 4 / float(self.numParameters + 4)                    # time constant for cumulation for covariance matrix
        self.cumStep = (self.muEff + 2) / (self.numParameters + self.muEff + 3)# t-const for cumulation for Size control
        self.muCov = self.muEff                   # size of mu used for calculating learning rate covLearningRate
        self.covLearningRate = ((1 / self.muCov) * 2 / (self.numParameters + 1.4) ** 2 + (1 - 1 / self.muCov) * # learning rate for
                 ((2 * self.muEff - 1) / ((self.numParameters + 2) ** 2 + 2 * self.muEff)))                       # covariance matrix
        self.dampings = 1 + 2 * max(0, sqrt((self.muEff - 1) / (self.numParameters + 1)) - 1) + self.cumStep
        # damping for stepSize usually close to 1 former damp == self.dampings/self.cumStep

        # Initialize dynamic (internal) strategy parameters and constants
        self.covPath = zeros(self.numParameters)
        self.stepPath = zeros(self.numParameters)                   # evolution paths for C and stepSize
        self.B = eye(self.numParameters, self.numParameters)         # B defines the coordinate system
        self.D = eye(self.numParameters, self.numParameters)         # diagonal matrix D defines the scaling
        self.C = dot(dot(self.B, self.D), dot(self.B, self.D).T)       # covariance matrix
        self.chiN = self.numParameters ** 0.5 * (1 - 1. / (4. * self.numParameters) + 1 / (21. * self.numParameters ** 2))
        # expectation of ||numParameters(0,I)|| == norm(randn(numParameters,1))

    def _learnStep(self):
        # Generate and evaluate lambda offspring
        arz = randn(self.numParameters, self.batchSize)
        arx = tile(self.center.reshape(self.numParameters, 1), (1, self.batchSize))\
                        + self.stepSize * dot(dot(self.B, self.D), arz)
        arfitness = zeros(self.batchSize)
        for k in range(self.batchSize):
            arfitness[k] = self._oneEvaluation(arx[:, k])

        # Sort by fitness and compute weighted mean into center
        arfitness, arindex = sorti(arfitness)  # minimization
        arz = arz[:, arindex]
        arx = arx[:, arindex]
        arzsel = arz[:, range(self.mu)]
        arxsel = arx[:, range(self.mu)]
        arxmut = arxsel - tile(self.center.reshape(self.numParameters, 1), (1, self.mu))

        zmean = dot(arzsel, self.weights)
        self.center = dot(arxsel, self.weights)

        if self.storeAllCenters:
            self._allCenters.append(self.center)

        # Cumulation: Update evolution paths
        self.stepPath = (1 - self.cumStep) * self.stepPath \
                + sqrt(self.cumStep * (2 - self.cumStep) * self.muEff) * dot(self.B, zmean)         # Eq. (4)
        hsig = norm(self.stepPath) / sqrt(1 - (1 - self.cumStep) ** (2 * self.numEvaluations / float(self.batchSize))) / self.chiN \
                    < 1.4 + 2. / (self.numParameters + 1)
        self.covPath = (1 - self.cumCov) * self.covPath + hsig * \
                sqrt(self.cumCov * (2 - self.cumCov) * self.muEff) * dot(dot(self.B, self.D), zmean) # Eq. (2)

        # Adapt covariance matrix C
        self.C = ((1 - self.covLearningRate) * self.C                    # regard old matrix   % Eq. (3)
             + self.covLearningRate * (1 / self.muCov) * (outer(self.covPath, self.covPath) # plus rank one update
                                   + (1 - hsig) * self.cumCov * (2 - self.cumCov) * self.C)
             + self.covLearningRate * (1 - 1 / self.muCov)                 # plus rank mu update
             * dot(dot(arxmut, diag(self.weights)), arxmut.T)
            )

        # Adapt step size self.stepSize
        self.stepSize *= exp((self.cumStep / self.dampings) * (norm(self.stepPath) / self.chiN - 1)) # Eq. (5)

        # Update B and D from C
        # This is O(n^3). When strategy internal CPU-time is critical, the
        # next three lines should be executed only every (alpha/covLearningRate/N)-th
        # iteration, where alpha is e.g. between 0.1 and 10
        self.C = (self.C + self.C.T) / 2 # enforce symmetry
        Ev, self.B = eig(self.C)          # eigen decomposition, B==normalized eigenvectors
        Ev = real(Ev)       # enforce real value
        self.D = diag(sqrt(Ev))      #diag(ravel(sqrt(Ev))) # D contains standard deviations now
        self.B = real(self.B)

        # convergence is reached
        if arfitness[0] == arfitness[-1] or (abs(arfitness[0] - arfitness[-1]) /
                                             (abs(arfitness[0]) + abs(arfitness[-1]))) <= self.stopPrecision:
            if self.verbose:
                print("Converged.")
            self.maxLearningSteps = self.numLearningSteps

        # or diverged, unfortunately
        if min(Ev) > 1e5:
            if self.verbose:
                print("Diverged.")
            self.maxLearningSteps = self.numLearningSteps

    @property
    def batchSize(self):
        return int(4 + floor(3 * log(self.numParameters)))


def sorti(vect):
    """ sort, but also return the indices-changes """
    tmp = sorted([(x_y[1], x_y[0]) for x_y in enumerate(ravel(vect))])
    res1 = array([x[0] for x in tmp])
    res2 = array([int(x[1]) for x in tmp])
    return res1, res2

