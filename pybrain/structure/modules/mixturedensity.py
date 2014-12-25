# $Id$
__author__ = 'Martin Felder'

from .neuronlayer import NeuronLayer
from pybrain.tools.functions import safeExp

    
class MixtureDensityLayer(NeuronLayer):
    """ Mixture of Gaussians output layer (Bishop 2006, Ch. 5.6) with diagonal
    covariance matrix. 

    Assumes inbuf contains K*3 neurons, with the first K giving the mixing 
    coefficients, the next K the standard deviations and the last K the means.
    """
    
    def __init__(self, dim, name = None, mix=5):
        """Initialize mixture density layer - mix gives the number of Gaussians
        to mix, dim is the dimension of the target(!) vector."""
        nUnits = mix * (dim + 2)  # mean vec + stddev and mixing coeff
        NeuronLayer.__init__(self, nUnits, name)
        self.nGaussians = mix
        self.nDims = dim
        
    def _forwardImplementation(self, inbuf, outbuf):
        """Calculate layer outputs (Gaussian parameters etc., not function 
        values!) from given activations """        
        K = self.nGaussians
        # Mixing parameters and stddevs
        outbuf[0:K*2] = safeExp(inbuf[0:K*2])
        outbuf[0:K] /= sum(outbuf[0:K])
        # Means
        outbuf[K*2:] = inbuf[K*2:]
        
    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        """Calculate the derivatives of output wrt. corresponding input 
        activations."""
        # Cannot calculate because we would need the targets!
        # ==> we just pass through the stuff from the trainer, who takes care 
        # of the rest
        inerr[:] = outerr
        
