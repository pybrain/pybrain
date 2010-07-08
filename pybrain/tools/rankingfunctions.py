""" Ranking functions that are used in Black-box optimization, or for selection. """

__author__ = 'Daan Wierstra and Tom Schaul'

from pybrain.utilities import Named
from random import randint
from scipy import zeros, argmax, array, power, exp, sqrt, var, zeros_like, arange, mean, log


def rankedFitness(R):
    """ produce a linear ranking of the fitnesses in R.

    (The highest rank is the best fitness)"""
    #l = sorted(list(enumerate(R)), cmp = lambda a,b: cmp(a[1],b[1]))
    #l = sorted(list(enumerate(l)), cmp = lambda a,b: cmp(a[1],b[1]))
    #return array(map(lambda (r, dummy): r, l))
    res = zeros_like(R)
    l = zip(R, range(len(R)))
    l.sort()
    for i, (_, j) in enumerate(l):
        res[j] = i
    return res


def normalizedFitness(R):
    return array((R - mean(R)) / sqrt(var(R))).flatten()


class RankingFunction(Named):
    """ Default: ranked and scaled to [0,1]."""

    def __init__(self, **args):
        self.setArgs(**args)
        n = self.__class__.__name__
        for k, val in args.items():
            n += '-' + str(k) + '=' + str(val)
        self.name = n

    def __call__(self, R):
        """ :key R: one-dimensional array containing fitnesses. """
        res = rankedFitness(R)
        return res / float(max(res))


class TournamentSelection(RankingFunction):
    """ Standard evolution tournament selection, the returned array contains intergers for the samples that
    are selected indicating how often they are. """

    tournamentSize = 2

    def __call__(self, R):
        res = zeros(len(R))
        for i in range(len(R)):
            l = [i]
            for dummy in range(self.tournamentSize - 1):
                randindex = i
                while randindex == i:
                    randindex = randint(0, len(R) - 1)
                l.append(randindex)
            fits = map(lambda x: R[x], l)
            res[argmax(fits)] += 1
        return res


class SmoothGiniRanking(RankingFunction):
    """ a smooth ranking function that gives more importance to examples with better fitness.

    Rescaled to be between 0 and 1"""

    gini = 0.1
    linearComponent = 0.

    def __call__(self, R):
        def smoothup(x):
            """ produces a mapping from [0,1] to [0,1], with a specific gini coefficient. """
            return power(x, 2 / self.gini - 1)
        ranks = rankedFitness(R)
        res = zeros(len(R))
        for i in range(len(ranks)):
            res[i] = ranks[i] * self.linearComponent + smoothup(ranks[i] / float(len(R) - 1)) * (1 - self.linearComponent)
        res /= max(res)
        return res


class ExponentialRanking(RankingFunction):
    """ Exponential transformation (with a temperature parameter) of the rank values. """

    temperature = 10.

    def __call__(self, R):
        ranks = rankedFitness(R)
        ranks = ranks / (len(R) - 1.0)
        return exp(ranks * self.temperature)

class HansenRanking(RankingFunction):
    """ Ranking, as used in CMA-ES """

    def __call__(self, R):
        ranks = rankedFitness(R)
        return array([max(0., x) for x in log(len(R)/2.+1.0)-log(len(R)-array(ranks))])


class TopSelection(RankingFunction):
    """ Select the fraction of the best ranked fitnesses. """

    topFraction = 0.1

    def __call__(self, R):
        res = zeros(len(R))
        ranks = rankedFitness(R)
        cutoff = len(R) * (1. - self.topFraction)
        for i in range(len(R)):
            if ranks[i] >= cutoff:
                res[i] = 1.0
            else:
                res[i] = 0.0
        return res


class TopLinearRanking(TopSelection):
    """ Select the fraction of the best ranked fitnesses
    and scale them linearly between 0 and 1.  """

    topFraction = 0.2

    def __call__(self, R):
        res = zeros(len(R))
        ranks = rankedFitness(R)
        cutoff = len(R) * (1. - self.topFraction)
        for i in range(len(R)):
            if ranks[i] >= cutoff:
                res[i] = ranks[i] - cutoff
            else:
                res[i] = 0.0
        res /= max(res)
        return res

    def getPossibleParameters(self, numberOfSamples):
        x = 1. / float(numberOfSamples)
        return arange(x * 2, 1 + x, x)

    def setParameter(self, p):
        self.topFraction = p


class BilinearRanking(RankingFunction):
    """ Bi-linear transformation, rescaled. """

    bilinearFactor = 20

    def __call__(self, R):
        ranks = rankedFitness(R)
        res = zeros(len(R))
        transitionpoint = 4 * len(ranks) / 5
        for i in range(len(ranks)):
            if ranks[i] < transitionpoint:
                res[i] = ranks[i]
            else:
                res[i] = ranks[i] + (ranks[i] - transitionpoint) * self.bilinearFactor
        res /= max(res)
        return res


