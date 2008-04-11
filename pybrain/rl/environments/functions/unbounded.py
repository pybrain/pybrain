__author__ = 'Tom Schaul, tom@idsia.ch'

from math import sqrt

from function import FunctionEnvironment


class UnboundedFunctionEnvironment(FunctionEnvironment):
    """ a function that does not have a minimum """    
    desiredValue = -1000
    
        
class ParabRFunction(UnboundedFunctionEnvironment):
    def f(self, x):
        return -x[0] + 100 * sum(x[1:]**2)
        
        
class SharpRFunction(UnboundedFunctionEnvironment):
    def f(self, x):
        return -x[0] + 100*sqrt(sum(x[1:]**2))