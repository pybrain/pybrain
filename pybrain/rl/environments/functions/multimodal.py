""" The functions implemented here are standard benchmarks from literature. """

__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import power, exp, cos, sqrt, rand, sin, floor, dot, ones, sign, randn, prod
from scipy.linalg import orth
from math import pi
from random import shuffle

from pybrain.rl.environments.functions.function import FunctionEnvironment
from pybrain.rl.environments.functions.transformations import penalize, generateDiags


class MultiModalFunction(FunctionEnvironment):
    """ A function with more than one local optima. """
    xdimMin = 2
    desiredValue = -1e-3

class FunnelFunction(MultiModalFunction):
    funnelSize = 1.0
    funnelDepth = 1.

    def f(self, x):
        return min(dot(x - 2.5 * ones(self.xdim), x - 2.5 * ones(self.xdim)), \
            self.funnelDepth * self.xdim + self.funnelSize * dot(x + 2.5 * ones(self.xdim), x + 2.5 * ones(self.xdim)))



class RastriginFunction(MultiModalFunction):
    """ A classical multimodal benchmark with plenty of local minima, globally arranged on a bowl. """
    def __init__(self, xdim=1, a=1, xopt=None):
        # additional parameter
        self.a = a
        FunctionEnvironment.__init__(self, xdim, xopt)

    def f(self, x):
        s = 0
        for i, xi in enumerate(x):
            ai = power(self.a, (i - 1) / (self.xdim - 1))
            s += (ai * xi) ** 2 - 10 * cos(2 * pi * ai * xi)
        return s + 10 * len(x)


class WeierstrassFunction(MultiModalFunction):
    """ Global optimum is not unique. 
    
    Standard setting: kmax = 20 (Other variants: kmax=11)."""
        
    kmax = 20
    
    def f(self, x):
        a = 0.5
        b = 3
        res = 0
        for k in range(self.kmax):
            res += sum(a ** k * cos(2 * pi * b ** k * (x + 0.5)))
            res -= self.xdim * a ** k * cos(2 * pi * b ** k * 0.5)
        return res


class SchaffersF7Function(MultiModalFunction):
        
    def f(self, x):
        s = sqrt(x[:-1] ** 2 + x[1:] ** 2)
        return sum(sqrt(s) * (1 + sin(50 * power(s, 0.2)) ** 2)) ** 2


class AckleyFunction(MultiModalFunction):
    def f(self, x):
        res = -20 * exp(-0.2 * sqrt(1. / self.xdim * sum(x ** 2)))
        res -= exp((1. / self.xdim) * sum(cos(2 * pi * x)))
        res += 20 + exp(1)
        return res


class GriewankFunction(MultiModalFunction):
    def f(self, x):
        prod = 1
        for i, xi in enumerate(x):
            prod *= cos(xi / sqrt(i + 1))
        return 1 + sum(x ** 2) / 4000. - prod
    
class BucheRastriginFunction(MultiModalFunction):
    """ Deceptive and highly multi-modal."""
    
    def f(self, x):
        z = x[:]
        for i in range(self.xdim):
            e = i/(self.xdim-1.)/2.
            if x[i] <= 0 or i%2==0:
                e += 1
            z[i] *= power(10, e)
        return dot(z,z) + 10 * self.xdim - 10*sum(cos(2*pi*z))
    
class GriewankRosenbrockFunction(MultiModalFunction):
    """ Composite between the two. """
    
    def f(self, x):
        s = 100 * (x[:-1] ** 2 - x[1:]) ** 2 + (x[:-1] - 1) ** 2
        return 1/(self.xdim-1.) * sum(s / 4000. - cos(s)) +1
    


class Schwefel_2_13Function(MultiModalFunction):
    def __init__(self, *args, **kwargs):
        MultiModalFunction.__init__(self, *args, **kwargs)
        self.A = floor(rand(self.xdim, self.xdim) * 200) - 100
        self.B = floor(rand(self.xdim, self.xdim) * 200) - 100
        self.alphas = 2 * pi * rand(self.xdim) - pi

    def f(self, x):
        res = 0
        for i in range(self.xdim):
            Ai = sum(self.A[i] * sin(self.alphas) + self.B[i] * cos(self.alphas))
            Bix = sum(self.A[i] * sin(x) + self.B[i] * cos(x))
            res += (Ai - Bix) ** 2
        return res
    
class Schwefel20Function(MultiModalFunction):
    """ f20 in BBOB. """
    
    penalized = True
    
    _k = 4.189828872724339
    _k2 = 4.20966874633/2
    
    def __init__(self, *args, **kwargs):
        MultiModalFunction.__init__(self, *args, **kwargs)
        self._signs = sign(randn(self.xdim))
        self._diags = generateDiags(10, self.xdim)
        self.xopt = self._k2 * self._signs
        
    def f(self, x):
        z = 2* x * self._signs        
        z[1:] += (z[:-1]-self.xopt[:-1]) * 0.25
        z = 100 * (dot(self._diags, (z-self.xopt)) + self.xopt)
        return - 1. / float(self.xdim) * sum(z * sin(sqrt(abs(z)))) + self._k + 100 * penalize(z / 100.)
        
    
class GallagherGauss101MeFunction(MultiModalFunction):
    """ 101 random local optima (medium peaks). """
    
    numPeaks = 101
    maxCond = 1000.
    optCond = sqrt(1000)

    def __init__(self, *args, **kwargs):
        MultiModalFunction.__init__(self, *args, **kwargs)
        print(self.numPeaks, self.xdim)
        self._opts = [(rand(self.xdim) - 0.5) * 8]
        self._opts.extend([(rand(self.xdim) - 0.5) * 9.8 for _ in range(self.numPeaks-1)])
        alphas = [power(self.maxCond, 2 * i / float(self.numPeaks - 2)) for i in range(self.numPeaks - 1)]
        shuffle(alphas)
        self._covs = [generateDiags(alpha, self.xdim, shuffled=True) / power(alpha, 0.25) for alpha in [self.optCond] + alphas]
        self._R = orth(rand(self.xdim, self.xdim))
        self._ws = [10] + [1.1 + 8 * i / float(self.numPeaks - 2) for i in range(self.numPeaks - 1)]
        
        
    def f(self, x):
        rxy = [dot(self._R, (x - o)) for o in self._opts]
        return (10 - max([self._ws[i] * exp(-1 / (2. * self.xdim) * dot(rxy[i], dot(self._covs[i], rxy[i]))) 
                          for i in range(self.numPeaks)])) ** 2


class GallagherGauss21HiFunction(GallagherGauss101MeFunction):
    """ 21 random local optima (high peaks). """
    
    numPeaks = 21
    optCond = 1000.


class KatsuuraFunction(MultiModalFunction):
    """ Has more than 10^dim optima. """
    
    def f(self, x):
        return - 1 + prod([power(1 + (i + 1) * sum([abs(2 ** j * x[i] - int(2 ** j * x[i])) * 2 ** -j 
                                                    for j in range(1, 33)]),
                                 10. / power(self.xdim, 1.2)) 
                          for i in range(self.xdim)])
    
class LunacekBiRastriginFunction(MultiModalFunction):
    """ A deceptive double-funnel structure with many local optima. 
    The bad funnel has about 70% of the search volume. """
    
    def __init__(self, *args, **kwargs):
        MultiModalFunction.__init__(self, *args, **kwargs)
        self._mu0 = 2.5
        self._s = 1 - 1 / (2 * sqrt(self.xdim + 20) - 8.2)
        self._mu1 = -sqrt((self._mu0 ** 2 - 1) / self._s)
        self._signs = sign(randn(self.xdim))
        self._R1 = orth(rand(self.xdim, self.xdim))
        self._R2 = orth(rand(self.xdim, self.xdim))
        self._diags = generateDiags(100, self.xdim)
        
    def f(self, x):
        x_ = x * self._signs * 2
        z = dot(self._R1, dot(self._diags, dot(self._R2, x_ - self._mu0)))
        return (min(dot(x_ - self._mu0, x_ - self._mu0),
                    self.xdim + self._s * dot(x_ - self._mu1, x_ - self._mu1)) 
                + 10 * (self.xdim - sum(cos(2 * pi * z))))

         
        

class BraninFunction(MultiModalFunction):
    """ Has 3 global optima at (-pi, 12.275), (pi, 2.275), (9.42478, 2.475) """

    xdimMax = 2

    _a = 1.
    _b = 5.1 / (4 * pi ** 2)
    _c = 5 / pi
    _d = 6.
    _e = 10.
    _f = 1. / (8 * pi)

    vopt = 0.397887

    _globalOptima = [[-pi, 12.275], [pi, 2.275], [9.42478, 2.475]]

    def f(self, x):
        return self._a * (x[1] - self._b * x[0] ** 2 + self._c * x[0] - self._d) ** 2 + self._e * ((1 - self._f) * cos(x[0]) + 1) - self.vopt

