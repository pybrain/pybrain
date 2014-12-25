__author__ = 'Tom Schaul, tom@idsia.ch'


import scipy

from pybrain.structure.modules.neuronlayer import NeuronLayer
from pybrain.tools.functions import safeExp


class SoftmaxLayer(NeuronLayer):
    """ A layer implementing a softmax distribution over the input."""

    # TODO: collapsing option?
    # CHECKME: temperature parameter?

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = safeExp(inbuf)
        outbuf /= sum(outbuf)

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:] = outerr


class PartialSoftmaxLayer(NeuronLayer):
    """Layer implementing a softmax distribution over slices of the input."""

    def __init__(self, size, slicelength):
        super(PartialSoftmaxLayer, self).__init__(size)
        self.slicelength = slicelength

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = safeExp(inbuf)
        outbuf.shape = scipy.size(outbuf) / self.slicelength, self.slicelength
        s = outbuf.sum(axis=1)
        outbuf = (outbuf.T / s).T.flatten()

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:] = outerr
