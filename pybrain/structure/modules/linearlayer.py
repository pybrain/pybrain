from neuronlayer import NeuronLayer
from pybrain.utilities import substitute

class LinearLayer(NeuronLayer):
    """ The simplest kind of module, not doing any transformation. """
    
    @substitute('pybrain.tools.pyrex._linearlayer.LinearLayer_forwardImplementation')
    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = inbuf
    
    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:] = outerr
        