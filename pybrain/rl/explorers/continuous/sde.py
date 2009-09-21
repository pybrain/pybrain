__author__ = "Thomas Rueckstiess, ruecksti@in.tum.de"

from scipy import random, ndarray, dot
from copy import copy

from pybrain.structure.modules.module import Module
from pybrain.rl.explorers.explorer import Explorer
from pybrain.tools.functions import expln, explnPrime
from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.structure import LinearLayer

class StateDependentExplorer(Explorer, ParameterContainer):
    """ A continuous explorer, that perturbs the resulting action with
        additive, normally distributed random noise. The exploration
        has parameter(s) sigma, which are related to the distribution's 
        standard deviation. In order to allow for negative values of sigma, 
        the real std. derivation is a transformation of sigma according
        to the expln() function (see pybrain.tools.functions).
    """
    
    def __init__(self, statedim, actiondim, sigma = 0.):
            Explorer.__init__(self, actiondim, actiondim)
            self.statedim = statedim
            self.actiondim = actiondim
            
            # initialize parameters to sigma
            ParameterContainer.__init__(self, actiondim, stdParams = 0)
            self.sigma = [sigma]*actiondim
            
            # exploration matrix (linear function)
            self.explmatrix = random.normal(0., expln(self.sigma), (statedim, actiondim))
            
            # store last state
            self.state = None

    def _setSigma(self, sigma):
        """ Wrapper method to set the sigmas (the parameters of the module) to a
            certain value. 
        """
        assert len(sigma) == self.actiondim
        self._params *= 0
        self._params += sigma

    def _getSigma(self):
        return self.params
    
    sigma = property(_getSigma, _setSigma)


    def activate(self, state, action):
        """ the super class commonly ignores the state and simply passes the
            action through the module. implement _forwardImplementation()
            in subclasses.
        """
        self.state = state
        print "sigma", self.sigma
        self.explmatrix = random.normal(0., expln(self.sigma), self.explmatrix.shape)
        return Module.activate(self, action)

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = inbuf + dot(self.state, self.explmatrix)

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        expln_sigma = expln(self.sigma)
        self._derivs += 0.000001
        inerr[:] = (outbuf - inbuf)
        
        # auto-alpha 
        # inerr /= expln_sigma**2
        # self._derivs /= expln_sigma**2
        
