__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import power, exp, cos, sqrt, rand, sin, floor
from math import pi

from function import FunctionEnvironment


class MultiModalFunction(FunctionEnvironment):
    
    xdimMin = 2
      

class RastriginFunction(MultiModalFunction):        
    
    def __init__(self, xdim = 1, a = 1, xopt = None):
        # additional parameter
        self.a = a
        FunctionEnvironment.__init__(self, xdim, xopt)
    
    def f(self, x):
        s = 0
        for i, xi in enumerate(x):
            ai = power(self.a, (i-1)/(self.xdim-1))
            s += (ai*xi)**2 - 10* cos(2*pi*ai*xi)
        return s + 10*len(x)
    
    
class WeierstrassFunction(MultiModalFunction):
    def f(self, x):
        a = 0.5
        b = 3
        kmax = 20
        res = 0
        for k in range(kmax):
            res += sum(a**k * cos(2*pi*b**k*(x+0.5)))
            res -= self.xdim * a**k * cos(2*pi*b**k * 0.5)
        return res
    
    
class AckleyFunction(MultiModalFunction):
    def f(self, x):
        res = -20 * exp(-0.2*sqrt(1./self.xdim*sum(x**2)))
        res -= exp((1./self.xdim) * sum(cos(2*pi*x)))
        res += 20+exp(1)
        return res
        
        
class GriewankFunction(MultiModalFunction):
    def f(self, x):
        prod = 1
        for i, xi in enumerate(x):
            prod *= cos(xi/sqrt(i+1))
        return 1 + sum(x**2)/4000. - prod
    
            
class Schwefel_2_13Function(MultiModalFunction):
    def __init__(self, *args, **kwargs):
        MultiModalFunction.__init__(self, *args, **kwargs)
        self.A = floor(rand(self.xdim, self.xdim)*200) - 100
        self.B = floor(rand(self.xdim, self.xdim)*200) - 100
        self.alphas = 2*pi*rand(self.xdim)-pi        
    
    def f(self, x):
        res = 0
        for i in range(self.xdim):
            Ai = sum(self.A[i] * sin(self.alphas) + self.B[i] * cos(self.alphas))
            Bix = sum(self.A[i] * sin(x) + self.B[i] * cos(x))
            res += (Ai-Bix)**2            
        return res
    
    