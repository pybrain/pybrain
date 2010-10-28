""" The functions implemented here are standard benchmarks from literature. """

__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import ones, sqrt, dot
from numpy.linalg.linalg import norm

from function import FunctionEnvironment


class SphereFunction(FunctionEnvironment):
    """ Simple quadratic function. """
    def f(self, x):
        return dot(x,x)


class SchwefelFunction(FunctionEnvironment):
    def f(self, x):
        s = 0
        for i in range(len(x)):
            s += sum(x[:i])**2
        return s


class CigarFunction(FunctionEnvironment):
    """ Bent Cigar function """
    xdimMin = 2

    def f(self, x):
        return x[0]**2 + 1e6*dot(x[1:],x[1:])


class TabletFunction(FunctionEnvironment):
    """ Also known as discus function."""
    xdimMin = 2

    def f(self, x):
        return 1e6*x[0]**2 + dot(x[1:],x[1:])


class ElliFunction(FunctionEnvironment):
    """ Ellipsoid. """
    def f(self, x):
        s = 0
        for i in range(len(x)):
            s += (x[i] * 1000**(i/(len(x)-1)))**2
        return s


class SharpRFunctionBis(FunctionEnvironment):
    """ Bounded version of the Sharp ridge function. """
    def f(self, x):
        return x[0]**2 + 100*sqrt(dot(x[1:],x[1:]))
    

class DiffPowFunction(FunctionEnvironment):
    """ Different powers.
    Standard setting: a=10 (other variants: a=4)
    """
    
    a = 10
    
    def f(self, x):
        s = 0
        for i in range(len(x)):
            s += abs(x[i])**(2+self.a*i/(len(x)-1))
        return s


class RosenbrockFunction(FunctionEnvironment):
    """ Banana-shaped function with a tricky optimum in the valley at 1,1.
    (has another, local optimum in higher dimensions)."""
    def __init__(self, xdim = 2, xopt = None):
        assert xdim >= self.xdimMin and not (self.xdimMax != None and xdim > self.xdimMax)
        self.xdim = xdim
        if xopt == None:
            self.xopt = ones(self.xdim)
        else:
            self.xopt = xopt
        self.reset()

    def f(self, x):
        return sum(100*(x[:-1]**2-x[1:])**2 + (x[:-1]-1)**2)
        
        
class GlasmachersFunction(FunctionEnvironment):
    """ Tricky! Designed to make most algorithms fail. """
    c = .1
    xdimMin = 2

    def f(self, x):
        m = self.xdim/2
        a = self.c * norm(x[:m])
        b = norm(x[m:])
        return a + b + sqrt(2*a*b+b**2)

