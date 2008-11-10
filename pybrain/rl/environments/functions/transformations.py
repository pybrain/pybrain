__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import rand, dot
from scipy.linalg import orth, norm

from function import FunctionEnvironment

    
class OppositeFunction(FunctionEnvironment):
    """ the opposite of a function """
    
    def __init__(self, basef):
        FunctionEnvironment.__init__(self, basef.xdim, basef.xopt)
        self.f = lambda x: -basef.f(x)


class TranslateFunction(FunctionEnvironment):
    """ change the position of the optimum """        
    
    def __init__(self, basef, distance = 0.1, offset = None):
        """ by default the offset is random, with a distance of 0.1 to the old one """
        FunctionEnvironment.__init__(self, basef.xdim, basef.xopt)
        if offset == None:
            offset = rand(basef.xdim)
            offset *= distance/norm(offset)
        self.xopt += offset
        if isinstance(basef, FunctionEnvironment):
            self.desiredValue = basef.desiredValue
        self.f =  lambda x: basef.f(x-offset)
    

class RotateFunction(FunctionEnvironment):
    """ make the dimensions non-seperable, by applying a matrix transformation to 
    x before it is given to the function """
    
    def __init__(self, basef, rotMat = None):
        """ by default the rotation matrix is random. """
        FunctionEnvironment.__init__(self, basef.xdim, basef.xopt)
        if rotMat == None:
            # make a random orthogonal rotation matrix
            self.M = orth(rand(basef.xdim, basef.xdim))
        else:
            self.M = rotMat
        if isinstance(basef, FunctionEnvironment):
            self.desiredValue = basef.desiredValue
        self.f = lambda x: basef.f(dot(x,self.M))
        
    
class CompositionFunction(FunctionEnvironment):
    """ composition of functions """
    # TODO

    