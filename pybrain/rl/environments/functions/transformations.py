__author__ = 'Tom Schaul, tom@idsia.ch'


from scipy import rand, dot, power, diag, eye, sqrt, sin, log, exp, ravel, clip, arange
from scipy.linalg import orth, norm, inv
from random import shuffle, random, gauss

from pybrain.rl.environments.functions.function import FunctionEnvironment
from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.rl.environments.fitnessevaluator import FitnessEvaluator
from pybrain.utilities import sparse_orth, dense_orth
from pybrain.rl.environments.functions.multiobjective import MultiObjectiveFunction


def oppositeFunction(basef):
    """ the opposite of a function """
    if isinstance(basef, FitnessEvaluator):
        if isinstance(basef, FunctionEnvironment):
            ''' added by JPQ '''
            if isinstance(basef, MultiObjectiveFunction):
                res = MultiObjectiveFunction()
            else:
            # ---
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
            self._offset = rand(basef.xdim)
            self._offset *= distance / norm(self._offset)
        else:
            self._offset = offset
        self.xopt += self._offset
        self.desiredValue = basef.desiredValue            
        self.toBeMinimized = basef.toBeMinimized
        def tf(x):
            if isinstance(x, ParameterContainer):
                x = x.params
            return basef.f(x - self._offset)
        self.f = tf
    

class RotateFunction(FunctionEnvironment):
    """ make the dimensions non-separable, by applying a matrix transformation to 
    x before it is given to the function """
    
    def __init__(self, basef, rotMat=None):
        """ by default the rotation matrix is random. """
        FunctionEnvironment.__init__(self, basef.xdim, basef.xopt)
        if rotMat == None:
            # make a random orthogonal rotation matrix
            self._M = orth(rand(basef.xdim, basef.xdim))
        else:
            self._M = rotMat
        self.desiredValue = basef.desiredValue            
        self.toBeMinimized = basef.toBeMinimized   
        self.xopt = dot(inv(self._M), self.xopt)
        def rf(x):
            if isinstance(x, ParameterContainer):
                x = x.params
            return basef.f(dot(x, self._M))    
        self.f = rf
        

def penalize(x, distance=5):
    ax = abs(x)
    tmp = clip(ax-distance, 0, ax.max())
    return dot(tmp, tmp)
    #return sum([max(0, abs(xi) - distance) ** 2 for xi in x])
        

class SoftConstrainedFunction(FunctionEnvironment):
    """ Soft constraint handling through a penalization term. """
    
    penalized = True
    
    def __init__(self, basef, distance=5, penalizationFactor=1.):
        FunctionEnvironment.__init__(self, basef.xdim, basef.xopt)
        self.desiredValue = basef.desiredValue            
        self.toBeMinimized = basef.toBeMinimized
        if basef.penalized:
            # already OK
            self.f = basef.f
        else:
            if not self.toBeMinimized:
                penalizationFactor *= -1
                
            def scf(x):
                if isinstance(x, ParameterContainer):
                    x = x.params
                return basef.f(x) + penalize(x, distance) * penalizationFactor
            
            self.f = scf
    
    
def generateDiags(alpha, dim, shuffled=False):    
    diags = [power(alpha, i / (2 * dim - 2.)) for i in range(dim)]
    if shuffled:
        shuffle(diags)
    return diag(diags)


class BBOBTransformationFunction(FunctionEnvironment):
    """ Reimplementation of the relatively complex set of function and 
    variable transformations, and their non-trivial combinations from BBOB 2010.
    But in clean, reusable code.
    """    
    
    def __init__(self, basef,
                 translate=True,
                 rotate=False,
                 conditioning=None,
                 asymmetry=None,
                 oscillate=False,
                 penalized=0,
                 desiredValue=1e-8,
                 gnoise=None,
                 unoise=None,
                 cnoise=None,
                 sparse=True,
                 ):
        FunctionEnvironment.__init__(self, basef.xdim, basef.xopt)
        self._name = basef.__class__.__name__
        self.desiredValue = desiredValue            
        self.toBeMinimized = basef.toBeMinimized
        
        if self.xdim < 500:
            sparse = False
        
        if sparse:
            try:
                from scipy.sparse import csc_matrix
            except:
                sparse = False
        
        if translate:            
            self.xopt = (rand(self.xdim) - 0.5) * 9.8
                    
        if conditioning:
            prefix = generateDiags(conditioning, self.xdim)                
            if sparse:
                prefix = csc_matrix(prefix)
                if rotate:
                    prefix = prefix * sparse_orth(self.xdim)
                    if oscillate or not asymmetry:
                        prefix = sparse_orth(self.xdim) * prefix                
            else:
                if rotate:
                    prefix = dot(prefix, dense_orth(self.xdim))
                    if oscillate or not asymmetry:
                        prefix = dot(dense_orth(self.xdim), prefix)
                
        elif rotate and asymmetry and not oscillate:
            if sparse:
                prefix = sparse_orth(self.xdim)
            else:
                prefix = dense_orth(self.xdim)
        elif sparse:
            prefix = None
        else:
            prefix = eye(self.xdim)  
            
        if penalized != 0:
            if self.penalized:
                penalized = 0
            else:
                self.penalized = True
        
        # combine transformations    
        if rotate:
            if sparse:
                r = sparse_orth(self.xdim)
                tmp1 = lambda x: ravel(x * r)
            else:
                r = dense_orth(self.xdim)
                tmp1 = lambda x: dot(x, r)
        else:
            tmp1 = lambda x: x
            
        if oscillate:
            tmp2 = lambda x: BBOBTransformationFunction.oscillatify(tmp1(x))     
        else:
            tmp2 = tmp1            
        
        if asymmetry is not None:
            tmp3 = lambda x: BBOBTransformationFunction.asymmetrify(tmp2(x), asymmetry)
        else:
            tmp3 = tmp2
            
        # noise
        ntmp = None
        if gnoise:
            ntmp = lambda f: f * exp(gnoise * gauss(0, 1))
        elif unoise:
            alpha = 0.49 * (1. / self.xdim) * unoise
            ntmp = lambda f: f * power(random(), unoise) * max(1, power(1e9 / (f + 1e-99), alpha * random())) 
        elif cnoise:
            alpha, beta = cnoise
            ntmp = lambda f: f + alpha * max(0, 1000 * (random() < beta) * gauss(0, 1) / (abs(gauss(0, 1)) + 1e-199))
            
        def noisetrans(f):
            if ntmp is None or f < 1e-8:
                return f
            else:
                return ntmp(f) + 1.01e-8
            
        if sparse:
            if prefix is None:
                tmp4 = lambda x: tmp3(x - self.xopt)
            else:
                tmp4 = lambda x: ravel(prefix * tmp3(x - self.xopt))
        else:
            tmp4 = lambda x: dot(prefix, tmp3(x - self.xopt))
                            
        self.f = lambda x: (noisetrans(basef.f(tmp4(x))) 
                            + penalized * penalize(x))
        

    @staticmethod
    def asymmetrify(x, beta=0.2):
        dim = len(x)
        return x * (x<=0) + (x>0) * exp((1+beta*arange(dim)/(dim-1.)*sqrt(abs(x))) * log(abs(x)+1e-100))
        #res = x.copy()
        #for i, xi in enumerate(x):
        #    if xi > 0:
        #        res[i] = power(xi, 1 + beta * i / (dim - 1.) * sqrt(xi))
        #return res
    
    @staticmethod
    def _oscillatify(x):
        if isinstance(x, float):
            res = [x]
        else:
            res = x.copy()        
        for i, xi in enumerate(res):
            if xi == 0: 
                continue
            elif xi > 0:
                s = 1 
                c1 = 10
                c2 = 7.9
            else:
                s = 1
                c1 = 5.5
                c2 = 3.1
            res[i] = s * exp(log(abs(xi)) + 0.049 * (sin(c1 * xi) + sin(c2 * xi)))
        if isinstance(x, float):
            return res[0]
        else:
            return res

    @staticmethod
    def oscillatify(x):
        return exp(log(abs(x)+1e-100)
                   + (x>0) * 0.049 * (sin(10 * x) + sin(7.9 * x))
                   + (x<0) * 0.049 * (sin(5.5 * x) + sin(3.1 * x)))
        