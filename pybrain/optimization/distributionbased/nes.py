__author__ = 'Daan Wierstra, Tom Schaul and Sun Yi'


from ves import VanillaGradientEvolutionStrategies
from pybrain.utilities import triu2flat, blockCombine
from scipy.linalg import inv, pinv2
from scipy import outer, dot, multiply, zeros, diag, mat, sum


class ExactNES(VanillaGradientEvolutionStrategies):
    """ A new version of NES, using the exact instead of the approximate
    Fisher Information Matrix, as well as a number of other improvements.
    (GECCO 2009).
    """

    # 4 kinds of baselines can be used:
    NOBASELINE = 0
    UNIFORMBASELINE = 1
    SPECIFICBASELINE = 2
    BLOCKBASELINE = 3

    #: Type of baseline. The most robust one is also the default.
    baselineType = BLOCKBASELINE

    learningRate = 1.

    def _calcBatchUpdate(self, fitnesses):
        samples = self.allSamples[-self.batchSize:]
        d = self.numParameters
        invA = inv(self.factorSigma)
        invSigma = inv(self.sigma)
        diagInvA = diag(diag(invA))

        # efficient computation of V, which corresponds to inv(Fisher)*logDerivs
        V = zeros((self.numDistrParams, self.batchSize))
        # u is used to compute the uniform baseline
        u = zeros(self.numDistrParams)
        for i in range(self.batchSize):
            s = dot(invA.T, (samples[i] - self.x))
            R = outer(s, dot(invA, s)) - diagInvA
            flatR = triu2flat(R)
            u[:d] += fitnesses[i] * (samples[i] - self.x)
            u[d:] += fitnesses[i] * flatR
            V[:d, i] += samples[i] - self.x
            V[d:, i] += flatR

        j = self.numDistrParams - 1
        D = 1 / invSigma[-1, -1]
        # G corresponds to the blocks of the inv(Fisher)
        G = 1 / (invSigma[-1, -1] + invA[-1, -1] ** 2)

        u[j] = dot(G, u[j])
        V[j, :] = dot(G, V[j, :])
        j -= 1
        for k in reversed(range(d - 1)):
            p = invSigma[k + 1:, k]
            w = invSigma[k, k]
            wg = w + invA[k, k] ** 2
            q = dot(D, p)
            c = dot(p, q)
            r = 1 / (w - c)
            rg = 1 / (wg - c)
            t = -(1 + r * c) / w
            tg = -(1 + rg * c) / wg

            G = blockCombine([[rg, tg * q],
                              [mat(tg * q).T, D + rg * outer(q, q)]])
            D = blockCombine([[r , t * q],
                              [mat(t * q).T, D + r * outer(q, q)]])
            u[j - (d - k - 1):j + 1] = dot(G, u[j - (d - k - 1):j + 1])
            V[j - (d - k - 1):j + 1, :] = dot(G, V[j - (d - k - 1):j + 1, :])
            j -= d - k


        # determine the update vector, according to different baselines.
        if self.baselineType == self.BLOCKBASELINE:
            update = zeros(self.numDistrParams)
            vsquare = multiply(V, V)
            j = self.numDistrParams - 1
            for k in reversed(range(self.numParameters)):
                b0 = sum(vsquare[j - (d - k - 1):j + 1, :], 0)
                b = dot(b0, fitnesses) / sum(b0)
                update[j - (d - k - 1):j + 1] = dot(V[j - (d - k - 1):j + 1, :], (fitnesses - b))
                j -= d - k
            b0 = sum(vsquare[:j + 1, :], 0)
            b = dot(b0, fitnesses) / sum(b0)
            update[:j + 1] = dot(V[:j + 1, :], (fitnesses - b))

        elif self.baselineType == self.SPECIFICBASELINE:
            update = zeros(self.numDistrParams)
            vsquare = multiply(V, V)
            for j in range(self.numDistrParams):
                b = dot(vsquare[j, :], fitnesses) / sum(vsquare[j, :])
                update[j] = dot(V[j, :], (fitnesses - b))

        elif self.baselineType == self.UNIFORMBASELINE:
            v = sum(V, 1)
            update = u - dot(v, u) / dot(v, v) * v

        elif self.baselineType == self.NOBASELINE:
            update = dot(V, fitnesses)

        else:
            raise NotImplementedError('No such baseline implemented')

        return update / self.batchSize



class OriginalNES(VanillaGradientEvolutionStrategies):
    """ Reference implementation of the original Natural Evolution Strategies algorithm (CEC-2008). """

    learningRate = 1.

    def _calcBatchUpdate(self, fitnesses):
        xdim = self.numParameters
        invSigma = inv(self.sigma)
        samples = self.allSamples[-self.batchSize:]
        phi = zeros((self.batchSize, self.numDistrParams + 1))
        phi[:, :xdim] = self._logDerivsX(samples, self.x, invSigma)
        phi[:, xdim:-1] = self._logDerivsFactorSigma(samples, self.x, invSigma, self.factorSigma)
        phi[:, -1] = 1

        update = dot(pinv2(phi), fitnesses)[:-1]
        return update

    def _logDerivsFactorSigma(self, samples, mu, invSigma, factorSigma):
        """ Compute the log-derivatives w.r.t. the factorized covariance matrix components.
        This implementation should be faster than the one in Vanilla. """
        res = zeros((len(samples), self.numDistrParams - self.numParameters))
        invA = inv(factorSigma)
        diagInvA = diag(diag(invA))

        for i, sample in enumerate(samples):
            s = dot(invA.T, (sample - mu))
            R = outer(s, dot(invA, s)) - diagInvA
            res[i] = triu2flat(R)
        return res

