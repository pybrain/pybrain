__author__ = 'Tom Schaul, tom@idsia.ch'

from neuronlayer import NeuronLayer
from pybrain.tools.functions import sigmoid
from pybrain.utilities import substitute


class SigmoidLayer(NeuronLayer):
    """Layer implementing the sigmoid squashing function."""

    @substitute('pybrain.pyrex._sigmoidlayer.SigmoidLayer_forwardImplementation')
    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = sigmoid(inbuf)
        
    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:] = outbuf * (1 - outbuf) * outerr
        
    @substitute('pybrain.pyrex._sigmoidlayer.SigmoidLayerforward')
    def forward(self, time=None):
        NeuronLayer.forward(self, time)