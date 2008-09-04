from neuronlayer import NeuronLayer
from module import Module
from pybrain.utilities import substitute

class LinearLayer(NeuronLayer):
    """ The simplest kind of module, not doing any transformation. """
    
    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = inbuf
    
    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:] = outerr