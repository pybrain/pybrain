""" The functions implemented here are standard benchmarks from literature. """

__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import power, exp, cos, sqrt, rand, sin, floor, dot, ones
from math import pi

from function import FunctionEnvironment



class MultiModalFunction(FunctionEnvironment):
    """ A function with more than one local optima. """
    xdimMin = 2    
    desiredValue = -1e-3
      
class FunnelFunction(MultiModalFunction):
    funnelSize = 1.0
    funnelDepth = 1.
    
    def f(self, x):
        return min( dot(x-2.5*ones(self.xdim), x-2.5*ones(self.xdim)), \
            self.funnelDepth * self.xdim + self.funnelSize * dot(x+2.5*ones(self.xdim), x+2.5*ones(self.xdim)) )



class RastriginFunction(MultiModalFunction):        
    """ A classical multimodal benchmark with plenty of local minima, globally arranged on a bowl. """
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
    
    
class BraninFunction(MultiModalFunction):
    """ Has 3 global optima at (-pi, 12.275), (pi, 2.275), (9.42478, 2.475) """
    
    xdimMax = 2
    
    _a = 1.
    _b = 5.1/(4*pi**2)
    _c = 5/pi
    _d = 6.
    _e = 10.
    _f = 1./(8*pi)
    
    vopt = 0.397887
    
    _globalOptima = [[-pi, 12.275], [pi, 2.275], [9.42478, 2.475]]
    
    def f(self, x):
        return self._a * (x[1]-self._b*x[0]**2+self._c*x[0]-self._d)**2 + self._e * ((1-self._f)*cos(x[0])+1) - self.vopt

