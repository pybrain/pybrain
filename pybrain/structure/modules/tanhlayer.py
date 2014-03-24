__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import tanh

from pybrain.structure.modules.neuronlayer import NeuronLayer


class TanhLayer(NeuronLayer):
    """ A layer implementing the tanh squashing function. """

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = tanh(inbuf)

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:] = (1 - outbuf**2) * outerr
