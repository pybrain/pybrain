""" Some multi-objective benchmark functions. 
Implemented according to the classical reference paper of Deb et al. (Evolutionary Computation 2002) """

from scipy import array, exp, sqrt, sin, cos, power
from pybrain.rl.environments.functions.function import FunctionEnvironment

__author__ = 'Tom Schaul, tom@idsia.ch'


class MultiObjectiveFunction(FunctionEnvironment):
    """ A function with multiple outputs. """
    ydim = 2 # by default
        
    @property
    def outdim(self):
        return self.ydim

class SchBenchmark(MultiObjectiveFunction):
    """ Schaffer 1987 """
    xdim = 1
    xdimMax = 1
    
    def f(self, x):
        return -array([x**2, (x-2)**2])
    
        
class FonBenchmark(MultiObjectiveFunction):
    """ Fonesca and Fleming 1993 """
    xdim = 3
    
    def f(self, x):
        f1 = 1 - exp(-sum((x-1/sqrt(3))**2))
        f2 = 1 - exp(-sum((x+1/sqrt(3))**2))
        return -array([f1, f2])
    
    
class PolBenchmark(MultiObjectiveFunction):
    """ Poloni 1997 """
    xdim = 2
    
    _A1 = 0.5 * sin(1) - 2*cos(1) + sin(2) -1.5*cos(2)
    _A2 = 1.5 * sin(1) - cos(1) + 2*sin(2) -0.5*cos(2)
    
    def f(self, x): 
        B1 = 0.5 * sin(x[0]) - 2*cos(x[0]) + sin(x[1]) -1.5*cos(x[1])
        B2 = 1.5 * sin(x[0]) - cos(x[0]) + 2*sin(x[1]) -0.5*cos(x[1])
    
        f1 = 1 + (self._A1-B1)**2 + (self._A2-B2)**2
        f2 = (x[0]+3)**2 + (x[1]+1)**2
        return -array([f1, f2])
    
    
class KurBenchmark(MultiObjectiveFunction):
    """ Kursawe 1990 """
    xdim = 3
    
    def f(self, x): 
        f1 = sum(-10*exp(-0.2*sqrt(x[:-1]**2+x[1:]**2)))        
        f2 = sum(power(abs(x), 0.8)+5*sin(x**3))
        return -array([f1, f2])
    
    
    
        