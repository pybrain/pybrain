__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from scipy import random, array, exp, log, asarray, clip, zeros, ravel, sqrt, dot
from neuronlayer import NeuronLayer
import pdb

class StateDependentLayer(NeuronLayer):
    
    def __init__(self, dim, module, name=None, onesigma=True):
        NeuronLayer.__init__(self, dim, name)
        self.exploration = zeros(dim, float)
        self.state = None
        self.onesigma = onesigma
        
        if self.onesigma:
            # one single parameter: sigma
            self.initParams(1, stdParams = 0)
        else:
            # sigmas for all parameters in the exploration module
            self.initParams(module.paramdim, stdParams = 0)

        # a module for the exploration
        assert module.outdim == dim
        self.module = module
        self.autoalpha = False
        self.enabled = True
        
    def setState(self, state):
        self.state = asarray(state)
        self.exploration[:] = self.module.activate(self.state)
        self.module.reset()
                
    def setSigma(self, sigma):
        """ wrapper method to set the sigmas (the parameters of the module) to a certain value. """
        assert len(sigma) == self.paramdim
        self.params *= 0
        self.params += sigma
        
    def drawRandomWeights(self):
        self.module._setParameters(random.normal(0, self.expln(self.params), self.module.paramdim)) 

    def _forwardImplementation(self, inbuf, outbuf):
        assert self.exploration != None
        if not self.enabled:
            outbuf[:] = inbuf
        else:
            outbuf[:] = inbuf + self.exploration
        self.exploration = zeros(self.dim, float)
        
    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        if self.onesigma:
            # algorithm for one global sigma for all mu's
            expln_params = self.expln(self.params)
            sumxsquared = dot(self.state, self.state)
            self.derivs += sum((outbuf - inbuf)**2 - expln_params**2 * sumxsquared) / expln_params * self.explnPrime(self.params)
            inerr[:] = (outbuf - inbuf)
        
            if not self.autoalpha:
                inerr /= expln_params**2 * sumxsquared
                self.derivs /= expln_params**2 * sumxsquared
        else:
            # algorithm for seperate sigma for each mu
            expln_params = self.expln(self.params).reshape(len(outbuf), len(self.state))
            explnPrime_params = self.explnPrime(self.params).reshape(len(outbuf), len(self.state))
        
            idx = 0
            for j in xrange(len(outbuf)):
                sigma_subst2 = dot(self.state**2,expln_params[j,:]**2) 
                for i in xrange(len(self.state)):
                    self.derivs[idx] = ((outbuf[j] - inbuf[j])**2 - sigma_subst2) / sigma_subst2 * \
                        self.state[i]**2*expln_params[j,i]*explnPrime_params[j,i]
                    if self.autoalpha:
                        self.derivs[idx] /= sigma_subst2
                    idx += 1
                inerr[j] = (outbuf[j] - inbuf[j])
                if not self.autoalpha:
                    inerr[j] /= sigma_subst2
                    
    
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
        return asarray(map(f, x))
    
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
        return asarray(map(f, x))
    