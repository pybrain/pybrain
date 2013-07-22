""" Some multi-objective benchmark functions.
Implemented according to the classical reference paper of Deb et al. (Evolutionary Computation 2002) """

from scipy import array, exp, sqrt, sin, cos, power, pi, arctan, ndarray
from pybrain.rl.environments.functions.function import FunctionEnvironment
from pybrain.structure.parametercontainer import ParameterContainer


__author__ = 'Tom Schaul, tom@idsia.ch'


class MultiObjectiveFunction(FunctionEnvironment):
    """ A function with multiple outputs. """
    ydim = 2 # by default
    
    feasible = None
    xbound = None
    constrained = None
    violation = None
    
    @property
    def outfeasible(self):
        return self.feasible
        
    @property
    def outviolation(self):
        return self.violation

    @property
    def outdim(self):
        return self.ydim
        
    def __call__(self, x):
        if isinstance(x, ParameterContainer):
            x = x.params
        assert type(x) == ndarray, 'FunctionEnvironment: Input not understood: '+str(type(x))
        res = self.f(x)
        return res

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

''' added by JPQ'''
class Deb(MultiObjectiveFunction):
    """ Deb 2001 """
    xdim = 2
    def __init__(self):
        self.xbound = []
        self.xbound.append((0.1,1.0))
        self.xbound.append((0.0,5.0))
        self.constrained = False

    def f(self, x):
        f1 = x[0]
        f2 = (1+x[1])/x[0]
        return -array([f1, f2])

class ConstDeb(MultiObjectiveFunction):
    """ Deb 2001 """
    xdim = 2
    def __init__(self):
        self.feasible = None
        self.xbound = []
        self.xbound.append((0.1,1.0))
        self.xbound.append((0.0,5.0))
        self.constrained = True

    def g(self, x):
        g1 = x[1] + 9.0*x[0] - 6.0
        g2 = -x[1] + 9.0*x[0] - 1.0
        if g1 >= 0 and g2 >= 0:
            return True,array([0.,0.])
        return False,array([g1,g2])
    def f(self, x):
        self.feasible,self.violation = self.g(x)
        ''' 
        not nice, due to the fact that oppositeFunction does not used
        the instance of this class for the evaluator but creates a new instance
        of class MultiObjectiveFunction
        '''
        MultiObjectiveFunction.feasible = self.feasible
        MultiObjectiveFunction.violation = self.violation
        f1 = x[0]
        f2 = (1+x[1])/x[0]
        return -array([f1, f2])

class Pol(MultiObjectiveFunction):
    """ Poloni 1997 """
    xdim = 2
    _A1 = 0.5 * sin(1) - 2*cos(1) + sin(2) -1.5*cos(2)
    _A2 = 1.5 * sin(1) - cos(1) + 2*sin(2) -0.5*cos(2)
    
    def __init__(self):
        self.xbound = []
        self.xbound.append((-pi,pi))
        self.xbound.append((-pi,pi))
        self.constrained = False


    def f(self, x):
        B1 = 0.5 * sin(x[0]) - 2*cos(x[0]) + sin(x[1]) -1.5*cos(x[1])
        B2 = 1.5 * sin(x[0]) - cos(x[0]) + 2*sin(x[1]) -0.5*cos(x[1])

        f1 = 1 + (self._A1-B1)**2 + (self._A2-B2)**2
        f2 = (x[0]+3)**2 + (x[1]+1)**2
        return -array([f1, f2])

class ConstSrn(MultiObjectiveFunction):
    """ Srinivas and Deb 1994 """
    xdim = 2
    def __init__(self):
        self.feasible = None
        self.xbound = []
        self.xbound.append((-20.0,20.0))
        self.xbound.append((-20.0,20.0))
        self.constrained = True

    def g(self, x):
        g1 = 225 - (x[1]**2 + x[0]**2)
        g2 = 3*x[1] -x[0] - 10.0
        if g1 >= 0 and g2 >= 0:
            return True,array([0.,0.])
        return False,array([g1,g2])
    def f(self, x):
        self.feasible,self.violation = self.g(x)
        MultiObjectiveFunction.feasible = self.feasible
        MultiObjectiveFunction.violation = self.violation
        f1 = 2+(x[0]-2)**2+(x[1]-1)**2
        f2 = 9*x[0]-(x[1]-1)**2
        return -array([f1, f2])
        
class ConstOsy(MultiObjectiveFunction):
    """ Osyczka and Kundu 1995 """
    xdim = 6
    def __init__(self):
        self.feasible = None
        self.xbound = []
        self.xbound.append((0.0,10.0))
        self.xbound.append((0.0,10.0))
        self.xbound.append((1.0,5.0))
        self.xbound.append((0.0,6.0))
        self.xbound.append((1.0,5.0))
        self.xbound.append((0.0,10.0))
        self.constrained = True

    def g(self, x):
        g1 = x[0] + x[1] -2.0
        g2 = 6.0 -x[0] -x[1]
        g3 = 2.0 -x[1] +x[0]
        g4 = 1.0 -x[0] +3.0*x[1]
        g5 = 4.0 -(x[2]-3)**2 -x[3]
        g6 = (x[4]-3)**2 +x[5] -4.0
        if g1 >= 0 and g2 >= 0 and g3 >= 0 and g4 >= 0 and g5 >= 0 and g6 >= 0:
            return True,array([0.,0.])
        return False,array([g1,g2,g3,g4,g5,g6])
    def f(self, x):
        self.feasible,self.violation = self.g(x)
        MultiObjectiveFunction.feasible = self.feasible
        MultiObjectiveFunction.violation = self.violation
        f1 = -(25.0*(x[0]-2.0)**2+(x[1]-2.0)**2+(x[2]-1.0)**2+(x[3]-4.0)**2+(x[4]-1.0)**2)
        f2 = x[0]**2+x[1]**2+x[2]**2+x[3]**2+x[4]**2+x[5]**2
        return -array([f1, f2])

class ConstTnk(MultiObjectiveFunction):
    """ Tanaka 1995 """
    xdim = 2
    def __init__(self):
        self.feasible = None
        self.xbound = []
        self.xbound.append((0.0,pi))
        self.xbound.append((0.0,pi))
        self.constrained = True

    def g(self, x):
        if x[1] == 0.0:
            A = pi/2.0
        else:
            A = arctan(x[0]/x[1])
        g1 = x[1]**2 + x[0]**2 - 1.0 -0.1*cos(16.0*A)
        g2 = 0.5 -(x[0]-0.5)**2 -(x[1]-0.5)**2 
        if g1 >= 0 and g2 >= 0:
            return True,array([0.,0.])
        return False,array([g1,g2])
    def f(self, x):
        self.feasible,self.violation = self.g(x)
        MultiObjectiveFunction.feasible = self.feasible
        MultiObjectiveFunction.violation = self.violation
        f1 = x[0]
        f2 = x[1]
        return -array([f1, f2])
    
class ConstBnh(MultiObjectiveFunction):
    """ Binh & Korn 1997 """
    xdim = 2
    def __init__(self):
        self.feasible = None
        self.xbound = []
        self.xbound.append((0.0,5.0))
        self.xbound.append((0.0,3.0))
        self.constrained = True

    def g(self, x):
        g1 = 25 - (x[0]-5.0)**2 - x[1]**2
        g2 = (x[0]-8.0)**2 + (x[1]+3.0)**2 - 7.7
        if g1 >= 0 and g2 >= 0:
            return True,array([0.,0.])
        return False,array([g1,g2])
    def f(self, x):
        self.feasible,self.violation = self.g(x)
        MultiObjectiveFunction.feasible = self.feasible
        MultiObjectiveFunction.violation = self.violation
        f1 = 4*x[0]**2 + 4*x[1]**2
        f2 = (x[0]-5)**2 + (x[1]-5)**2
        return -array([f1, f2])
# ---