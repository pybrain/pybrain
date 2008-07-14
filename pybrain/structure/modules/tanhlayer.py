__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import tanh

from neuronlayer import NeuronLayer
from pybrain.utilities import substitute


class TanhLayer(NeuronLayer):
    """ A layer implementing the sigmoid squashing function. """
    
    @substitute('pybrain.pyrex._tanhlayer.TanhLayer_forwardImplementation')
    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = tanh(inbuf)
        
    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:] = (1-outbuf**2)*outerr
        
    @substitute('pybrain.pyrex._tanhlayer.TanhLayerforward')
    def forward(self, time = None):
        NeuronLayer.forward(self, time)