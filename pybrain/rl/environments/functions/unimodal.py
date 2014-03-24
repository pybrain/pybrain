""" The functions implemented here are standard benchmarks from literature. """

__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import ones, sqrt, dot, sign, randn, power, rand, floor, array
from scipy.linalg import norm, orth

from pybrain.rl.environments.functions.function import FunctionEnvironment


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
        
    a = 1000
    
    def __init__(self, *args, **kwargs):
        FunctionEnvironment.__init__(self, *args, **kwargs)
        self._as = array([power(self.a, 2*i/(self.xdim-1.)) for i in range(self.xdim)])
        
    def f(self, x):
        return dot(self._as*x, x)

class StepElliFunction(ElliFunction):
    """ Plateaus make for a zero derivative """
    
    a = 10
    
    def __init__(self, *args, **kwargs):
        ElliFunction.__init__(self, *args, **kwargs)
        self._R = orth(rand(self.xdim, self.xdim))
    
    def f(self, x):
        # rounding
        z = floor(0.5 + x) * (x>0.5) + floor(0.5 + 10*x)/10. * (x<=0.5)
        z = dot(self._R, z)         
        return 0.1 * max(1e-4 * abs(z[0]), ElliFunction.f(self, z))
    
    
    
class AttractiveSectorFunction(FunctionEnvironment):
    """ Only one of the quadrants has the good values. """
    
    def __init__(self, *args, **kwargs):
        FunctionEnvironment.__init__(self, *args, **kwargs)
        self.xopt = (rand(self.xdim) - 0.5) * 9.8   
    
    def f(self, x):
        from transformations import BBOBTransformationFunction
        quad = (x*self.xopt > 0)
        sz = 100 * x * quad + x * (quad==False)         
        return power(BBOBTransformationFunction.oscillatify(dot(sz, sz)), 0.9)
        
        

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
    
    
class BoundedLinear(FunctionEnvironment):
    """ Linear function within the domain [-5,5], constant from the 
    best corner onwards. """
    
    def __init__(self, *args, **kwargs):
        FunctionEnvironment.__init__(self, *args, **kwargs)
        self._w = [power(10, i/(self.xdim-1.)) for i in range(self.xdim)]        
        self._signs = sign(randn(self.xdim))       
    
    def f(self, x):
        x_ = x[:]
        for i, xi in enumerate(x):
            if xi*self._signs[i] > 5:
                x_[i] = self._signs[i]*5
        return 5*sum(self._w) - dot(self._w*self._signs, x_) 
        
        
