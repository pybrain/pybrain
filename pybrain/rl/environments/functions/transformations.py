from pybrain.rl.environments.fitnessevaluator import FitnessEvaluator
__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import rand, dot
from scipy.linalg import orth, norm, inv

from function import FunctionEnvironment
from pybrain.structure.parametercontainer import ParameterContainer


def oppositeFunction(basef):
    """ the opposite of a function """
    if isinstance(basef, FitnessEvaluator):
        if isinstance(basef, FunctionEnvironment):
            res = FunctionEnvironment(basef.xdim, basef.xopt)
        else:
            res = FitnessEvaluator()        
        res.f = lambda x:-basef.f(x)
        if not basef.desiredValue is None:
            res.desiredValue = -basef.desiredValue
        res.toBeMinimized = not basef.toBeMinimized
        return res
    else:    
        return lambda x:-basef(x)
            

class TranslateFunction(FunctionEnvironment):
    """ change the position of the optimum """        
    
    def __init__(self, basef, distance=0.1, offset=None):
        """ by default the offset is random, with a distance of 0.1 to the old one """
        FunctionEnvironment.__init__(self, basef.xdim, basef.xopt)
        if offset == None:
            offset = rand(basef.xdim)
            offset *= distance / norm(offset)
        self.xopt += offset
        self.desiredValue = basef.desiredValue            
        self.toBeMinimized = basef.toBeMinimized
        def tf(x):
            if isinstance(x, ParameterContainer):
                x = x.params
            return basef.f(x - offset)
        self.f = tf
    

class RotateFunction(FunctionEnvironment):
    """ make the dimensions non-separable, by applying a matrix transformation to 
    x before it is given to the function """
    
    def __init__(self, basef, rotMat=None):
        """ by default the rotation matrix is random. """
        FunctionEnvironment.__init__(self, basef.xdim, basef.xopt)
        if rotMat == None:
            # make a random orthogonal rotation matrix
            self.M = orth(rand(basef.xdim, basef.xdim))
        else:
            self.M = rotMat
        self.desiredValue = basef.desiredValue            
        self.toBeMinimized = basef.toBeMinimized   
        self.xopt = dot(inv(self.M), self.xopt)
        def rf(x):
            if isinstance(x, ParameterContainer):
                x = x.params
            return basef.f(dot(x, self.M))    
        self.f = rf
        
    
class CompositionFunction(FunctionEnvironment):
    """ composition of functions """
    # TODO

    
