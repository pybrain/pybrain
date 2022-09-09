__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.structure.modules.neuronlayer import NeuronLayer

class ReluLayer(NeuronLayer):
    """ Layer of rectified linear units (relu). """

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = inbuf * (inbuf > 0)

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:] = outerr * (inbuf > 0)