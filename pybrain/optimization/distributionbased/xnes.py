__author__ = 'Tom Schaul, Sun Yi, Tobias Glasmachers'


from pybrain.tools.rankingfunctions import HansenRanking
from pybrain.optimization.distributionbased.distributionbased import DistributionBasedOptimizer
from pybrain.auxiliary.importancemixing import importanceMixing
from scipy.linalg import expm2
from scipy import dot, array, randn, eye, outer, exp, trace, floor, log, sqrt


class XNES(DistributionBasedOptimizer):
    """ NES with exponential parameter representation. """

    # parameters, which can be set but have a good (adapted) default value
    covLearningRate = None
    centerLearningRate = 1.0
    scaleLearningRate = None
    uniformBaseline = True
    batchSize = None
    shapingFunction = HansenRanking()
    importanceMixing = False
    forcedRefresh = 0.01

    # fixed settings
    mustMaximize = True
    storeAllEvaluations = True
    storeAllEvaluated = True
    storeAllDistributions = False

    def _additionalInit(self):
        # good heuristic default parameter settings
        dim = self.numParameters
        if self.covLearningRate is None:
            self.covLearningRate = 0.6*(3+log(dim))/dim/sqrt(dim)
        if self.scaleLearningRate is None:
            self.scaleLearningRate = self.covLearningRate
        if self.batchSize is None:
            if self.importanceMixing:
                self.batchSize = 10*dim
            else:
                self.batchSize = 4+int(floor(3*log(dim)))

        # some bookkeeping variables
        self._center = self._initEvaluable.copy()
        self._A = eye(self.numParameters) # square root of covariance matrix
        self._invA = eye(self.numParameters)
        self._logDetA = 0.
        self._allPointers = []
        self._allGenSteps = [0]
        if self.storeAllDistributions:
            self._allDistributions = [(self._center.copy(), self._A.copy())]

    def _learnStep(self):
        """ Main part of the algorithm. """
        I = eye(self.numParameters)
        self._produceSamples()
        utilities = self.shapingFunction(self._currentEvaluations)
        utilities /= sum(utilities)  # make the utilities sum to 1
        if self.uniformBaseline:
            utilities -= 1./self.batchSize
        samples = array(map(self._base2sample, self._population))

        dCenter = dot(samples.T, utilities)
        covGradient = dot(array([outer(s,s) - I for s in samples]).T, utilities)
        covTrace = trace(covGradient)
        covGradient -= covTrace/self.numParameters * I
        dA = 0.5 * (self.scaleLearningRate * covTrace/self.numParameters * I
                    +self.covLearningRate * covGradient)

        self._lastLogDetA = self._logDetA
        self._lastInvA = self._invA

        self._center += self.centerLearningRate * dot(self._A, dCenter)
        self._A = dot(self._A, expm2(dA))
        self._invA = dot(expm2(-dA), self._invA)
        self._logDetA += 0.5 * self.scaleLearningRate * covTrace
        if self.storeAllDistributions:
            self._allDistributions.append((self._center.copy(), self._A.copy()))

    @property
    def _lastA(self): return self._allDistributions[-2][1]
    @property
    def _lastCenter(self): return self._allDistributions[-2][0]
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
            fits = map(lambda x:-x, fits)
        return fits

    def _produceSample(self):
        return randn(self.numParameters)

    def _sample2base(self, sample):
        """ How does a sample look in the outside (base problem) coordinate system? """
        return dot(self._A, sample)+self._center

    def _base2oldsample(self, e):
        """ How would the point have looked in the previous reference coordinates? """
        return dot(self._lastInvA, (e - self._lastCenter))

    def _base2sample(self, e):
        """ How does the point look in the present one reference coordinates? """
        return dot(self._invA, (e - self._center))

    def _oldpdf(self, s):
        s = self._base2oldsample(self._sample2base(s))
        return exp(-0.5*dot(s,s)- self._lastLogDetA)

    def _newpdf(self, s):
        return exp(-0.5*dot(s,s)- self._logDetA)

    def _produceSamples(self):
        """ Append batch size new samples and evaluate them. """
        reuseindices = []
        if self.numLearningSteps == 0 or not self.importanceMixing:
            [self._oneEvaluation(self._sample2base(self._produceSample())) for _ in range(self.batchSize)]
            self._pointers = list(range(len(self._allEvaluated)-self.batchSize, len(self._allEvaluated)))
        else:
            reuseindices, newpoints = importanceMixing(map(self._base2sample, self._currentEvaluations),
                                                       self._oldpdf, self._newpdf, self._produceSample, self.forcedRefresh)
            [self._oneEvaluation(self._sample2base(s)) for s in newpoints]
            self._pointers = ([self._pointers[i] for i in reuseindices]+
                              range(len(self._allEvaluated)-self.batchSize+len(reuseindices), len(self._allEvaluated)))
        self._allGenSteps.append(self._allGenSteps[-1]+self.batchSize-len(reuseindices))
        self._allPointers.append(self._pointers)



if __name__ == '__main__':
    from pybrain.rl.environments.functions.unimodal import RosenbrockFunction
    from scipy import ones
    dim = 10
    f = RosenbrockFunction(dim)
    l = XNES(f, -ones(dim))
    print(l.learn())
    print('Evaluations needed:', len(l._allEvaluations))
