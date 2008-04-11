__author__ = 'Tom Schaul, tom@idsia.ch'


from scipy import ones

from function import FunctionEnvironment
    

class SphereFunction(FunctionEnvironment):
    def f(self, x):
        return sum(x**2)


class SchwefelFunction(FunctionEnvironment):
    def f(self, x):
        s = 0
        for i in range(len(x)):
            s += sum(x[:i])**2
        return s
    

class CigarFunction(FunctionEnvironment):
    xdimMin = 2
    
    def f(self, x):
        return x[0]**2 + 1e6*sum(x[1:]**2)


class TabletFunction(FunctionEnvironment):
    xdimMin = 2
    
    def f(self, x):
        return 1e6*x[0]**2 + sum(x[1:]**2)
                                            

class ElliFunction(FunctionEnvironment):
    def f(self, x):
        s = 0
        for i in range(len(x)):
            s += (x[i] * 1000**(i/(len(x)-1)))**2
        return s        
        
                
class DiffPowFunction(FunctionEnvironment):
    def f(self, x):
        s = 0
        for i in range(len(x)):
            s += abs(x[i])**(2+10*i/(len(x)-1))
        return s
    
    
class RosenbrockFunction(FunctionEnvironment):
    
    def __init__(self, xdim = 2, xopt = None):
        assert xdim >= self.xdimMin and not (self.xdimMax != None and xdim > self.xdimMax)
        self.xdim = xdim
        if xopt == None:
            self.xopt = ones(self.xdim)
        else:
            self.xopt = xopt
        self.reset()
    
    def f(self, x):
        s = 0
        for i in range(len(x)-1):
            s += 100 * (x[i]**2 - x[i+1])**2 + (x[i]-1)**2
        return s