from neuronlayer import NeuronLayer
from module import Module
from pybrain.utilities import substitute

class LinearLayer(NeuronLayer):
    """ The simplest kind of module, not doing any transformation. """
    
    @substitute('pybrain.pyrex._linearlayer.LinearLayer_forwardImplementation')
    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = inbuf
    
    @substitute('pybrain.pyrex._linearlayer.LinearLayer_backwardImplementation')
    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:] = outerr
        
    
    @substitute('pybrain.pyrex._linearlayer.LinearLayerforward')
    def forward(self, time = None): 
        Module.forward(self, time)
    
    @substitute('pybrain.pyrex._linearlayer.LinearLayerbackward')
    def backward(self, time = None): 
        Module.backward(self, time)
        