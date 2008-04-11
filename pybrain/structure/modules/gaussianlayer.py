__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from scipy import random, array, exp, log
from neuronlayer import NeuronLayer


class GaussianLayer(NeuronLayer):
    """ A layer implementing a gaussian interpretation of the input. The mean is the input, 
        the sigmas are stored in the module parameters. """

    def expln(self, x):
        """ This function ensures that the values of the array are always positive. It is 
            ln(x+1)+1 for x=>0 and exp(x) for x<0. """
        def f(val):
            if val<0:
                # exponential function for x<0
                return exp(val)
            else:
                # natural log function (slightly shifted) for x>=0
                return log(val+1.0)+1
        return array(map(f, x))
    
    def explnPrime(self, x):
        """ This function is the first derivative of the self.expln function (above).
            It is needed for the backward pass of the module. """
        def f(val):
            if val<0:
                # exponential function for x<0
                return exp(val)
            else:
                # linear function for x>=0
                return 1.0/(val+1.0)
        return array(map(f, x))
    
    
    def __init__(self, dim, name=None):
        NeuronLayer.__init__(self, dim, name)
        # initialize sigmas to 0
        self.initParams(dim, stdParams = 0)
        # if autoalpha is set to True, alpha_sigma = alpha_mu = alpha*sigma^2
        self.autoalpha = False
    
    def setSigma(self, sigma):
        """ wrapper method to set the sigmas (the parameters of the module) to a certain value. """
        assert len(sigma) == self.indim
        self.params *= 0
        self.params += sigma
    
    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = random.normal(inbuf, self.expln(self.params))
        
    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        expln_params = self.expln(self.params)
        self.derivs += ((outbuf - inbuf)**2 - expln_params**2) / expln_params * self.explnPrime(self.params)
        inerr[:] = (outbuf - inbuf)
        
        if not self.autoalpha:
            inerr /= expln_params**2
            self.derivs /= expln_params**2