__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.structure.modules.neuronlayer import NeuronLayer
from pybrain.tools.functions import sigmoid


class SigmoidLayer(NeuronLayer):
    """Layer implementing the sigmoid squashing function."""

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = sigmoid(inbuf)

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:] = outbuf * (1 - outbuf) * outerr

