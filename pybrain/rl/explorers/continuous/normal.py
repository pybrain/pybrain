__author__ = "Thomas Rueckstiess, ruecksti@in.tum.de"

from scipy import random, ndarray

from pybrain.rl.explorers.explorer import Explorer
from pybrain.tools.functions import expln


class NormalExplorer(Explorer):
    """ A continuous explorer, that perturbs the resulting action with
        additive, normally distributed random noise. The exploration
        has a parameter sigma, which is related to the distribution's 
        standard deviation. In order to allow for negative values of sigma, 
        the real std. derivation is a transformation of sigma according
        to the expln() function (see pybrain.tools.functions).
    """
    
    def __init__(self, sigma = 2., decay=None, covariance='diagonal'):
        """ takes an initial value for sigma, an optional decay multiplier
            and a covariance choice, which can be one of:
                - diagonal (default): n-dimensional vector representing the
                  diagonal of the covariance matrix
                - spherical: one single value multiplied by Eye(n) to get the
                  covariance matrix
                - full: the full covariance matrix
        """
        assert covariance in ['diagonal', 'spherical', 'full'], \
            'unknown covariance type: %s'%covariance
        
        self.covariance = covariance
        self.sigma = sigma
        self.decay = decay
    
    
    def activate(self, state, action):
        """ 
        """
        
        if self.covariance == 'full':
            assert isinstance(self.sigma, ndarray), 'sigma is not a covariance matrix'
            assert self.sigma.shape[0] == self.sigma.shape[1] == len(action), \
                'shape mismatch for covariance matrix'
        
            r = random.multivariate_normal([0]*len(action), expln(sigma))
        
        else:
            if self.covariance == 'diagonal':
                assert len(self.sigma) == len(action), 'shape mismatch for sigma vector'
            
            r = random.normal(0, self.sigma, len(action))

        action += r
        
        if self.decay != None:
            self.sigma *= self.decay
        
        return action
        