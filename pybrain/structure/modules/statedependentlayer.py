__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from scipy import random, asarray, zeros, dot

from neuronlayer import NeuronLayer
from pybrain.tools.functions import expln, explnPrime
from pybrain.structure.parametercontainer import ParameterContainer

class StateDependentLayer(NeuronLayer, ParameterContainer):
    
    def __init__(self, dim, module, name=None, onesigma=True):
        NeuronLayer.__init__(self, dim, name)
        self.exploration = zeros(dim, float)
        self.state = None
        self.onesigma = onesigma
        
        if self.onesigma:
            # one single parameter: sigma
            ParameterContainer.__init__(self, 1, stdParams = 0)
        else:
            # sigmas for all parameters in the exploration module
            ParameterContainer.__init__(self, module.paramdim, stdParams = 0)

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
        self.module._setParameters(random.normal(0, expln(self.params), self.module.paramdim)) 

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
            expln_params = expln(self.params)
            sumxsquared = dot(self.state, self.state)
            self.derivs += sum((outbuf - inbuf)**2 - expln_params**2 * sumxsquared) / expln_params * explnPrime(self.params)
            inerr[:] = (outbuf - inbuf)
        
            if not self.autoalpha:
                inerr /= expln_params**2 * sumxsquared
                self.derivs /= expln_params**2 * sumxsquared
        else:
            # algorithm for seperate sigma for each mu
            expln_params = expln(self.params).reshape(len(outbuf), len(self.state))
            explnPrime_params = explnPrime(self.params).reshape(len(outbuf), len(self.state))
        
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
