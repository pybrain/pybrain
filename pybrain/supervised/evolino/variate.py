__author__ = 'Michael Isik'

from random import uniform, random, gauss
from numpy  import tan, pi


class UniformVariate:
    def __init__(self, min_val=0., max_val=1.):
        """ Initializes the uniform variate with a min and a max value.
        """
        self._min_val = min_val
        self._max_val = max_val

    def getSample(self, min_val=None, max_val=None):
        """ Returns a random value between min_val and max_val.
        """
        if min_val is None: min_val = self._min_val
        if max_val is None: max_val = self._max_val
        return uniform(min_val, max_val)

class CauchyVariate:
    def __init__(self, x0=0., alpha=1.):
        """ :key x0: Median and mode of the Cauchy distribution
            :key alpha: scale
        """
        self.x0 = x0
        self.alpha = alpha

    def getSample(self, x0=None, alpha=None):
        if x0    is None: x0 = self.x0
        if alpha is None: alpha = self.alpha
        uniform_variate = random()
        cauchy_variate = x0 + alpha * tan(pi * (uniform_variate - 0.5))
        return cauchy_variate


class GaussianVariate:
    def __init__(self, x0=0., alpha=1.):
        """ :key x0: Mean
            :key alpha: standard deviation
        """
        self.x0 = x0
        self.alpha = alpha

    def getSample(self, x0=None, alpha=None):
        if x0    is None: x0 = self.x0
        if alpha is None: alpha = self.alpha
        return gauss(x0, alpha)


