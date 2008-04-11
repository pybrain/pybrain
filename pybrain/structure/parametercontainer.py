__author__ = 'Daan Wierstra and Tom Schaul'

from scipy import size, zeros
from numpy.random import randn


class ParameterContainer(object):
    """ A common interface implemented by all classes which
    contains data that can change during execution (i.e. trainable parameters)
    and should be losslessly storable and retrievable to files """
    
    params = None
    derivs = None
    paramdim = 0
    
    # if this variable is set, than only the owner can set the params or the derivs of the container
    owner = None
    
    def _setParameters(self, p, owner = None):
        """ @param p: an array of numbers """
        assert self.owner == owner
        self.params = p
        self.paramdim = size(self.params)
        
    def getParameters(self):
        """ @rtype: an array of numbers. """
        return self.params
                
    def _setDerivatives(self, d, owner = None):
        """ @param d: an array of numbers of self.paramdim """
        assert size(d) == self.paramdim
        assert self.owner == owner
        self.derivs = d
    
    def resetDerivatives(self):
        """ @note: this method only sets the values to zero, it does not initialize the array. """
        assert self.derivs != None
        self.derivs[:] = zeros(self.paramdim)
    
    def getDerivatives(self):
        """ @rtype: an array of numbers. """
        return self.derivs
    
    def initParams(self, dim, stdParams = 1.):
        """ initialize all parameters with random values, normally distributed around 0
            @param stdParams: standard deviation of the values (default: 1). 
        """
        self.paramdim = dim
        self.params = randn(self.paramdim)*stdParams
        self.derivs = zeros(self.paramdim)