# -*- coding: utf-8 -*-


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'
__version__ = '$Id$'


from pybrain.structure.modules.module import Module
from pybrain.structure.modules.neuronlayer import NeuronLayer
from pybrain.tools.functions import sigmoid, sigmoidPrime


class MultiplicationLayer(NeuronLayer):
    """Layer that implements pairwise multiplication."""

    def __init__(self, dim, name=None):
        Module.__init__(self, 2 * dim, dim, name)
        self.setArgs(dim=dim, name=self.name)

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf += inbuf[:self.outdim] * inbuf[self.outdim:]

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:self.outdim] += inbuf[self.outdim:] * outerr
        inerr[self.outdim:] += inbuf[:self.outdim] * outerr


class GateLayer(NeuronLayer):
    """Layer that implements pairwise input multiplication, with one element of
    the pair being squashed.

    If a GateLayer of size n is created, it will have 2 * n inputs and n
    outputs. The i'th output is calculated as sigmoid(I_i) * I_(i + n) where I
    is the vector of inputs."""

    def __init__(self, dim, name=None):
        Module.__init__(self, 2 * dim, dim, name)
        self.setArgs(dim=dim, name=self.name)

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf += sigmoid(inbuf[:self.outdim]) * inbuf[self.outdim:]

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:self.outdim] += (sigmoidPrime(inbuf[:self.outdim])
                                * inbuf[self.outdim:]
                                * outerr)
        inerr[self.outdim:] += (sigmoid(inbuf[:self.outdim])
                                * outerr)


class DoubleGateLayer(NeuronLayer):
    """Layer that implements a continuous if-then-else.

    If a DoubleGateLayer of size n is created, it will have 2 * n inputs and
    2 * n outputs. The i'th output is calculated as sigmoid(I_i) * I_(i + n) for
    i < n and as (1 - sigmoid(I_i) * I_(i + n) for i >= n where I is the vector
    of inputs."""

    def __init__(self, dim, name=None):
        Module.__init__(self, 2 * dim, 2 * dim, name)
        self.setArgs(dim=dim, name=self.name)

    def _forwardImplementation(self, inbuf, outbuf):
        dim = self.indim / 2
        outbuf[:dim] += sigmoid(inbuf[:dim]) * inbuf[dim:]
        outbuf[dim:] += (1 - sigmoid(inbuf[:dim])) * inbuf[dim:]

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        dim = self.indim / 2
        in0 = inbuf[:dim]
        in1 = inbuf[dim:]
        out0 = outerr[:dim]
        out1 = outerr[dim:]
        inerr[:dim] += sigmoidPrime(in0) * in1 * out0
        inerr[dim:] += sigmoid(in0) * out0

        inerr[:dim] -= sigmoidPrime(in0) * in1 * out1
        inerr[dim:] += (1 - sigmoid(in0)) * out1


class SwitchLayer(NeuronLayer):
    """Layer that implements pairwise multiplication."""
    #:TODO: Misleading docstring

    def __init__(self, dim, name=None):
        Module.__init__(self, dim, dim * 2, name)
        self.setArgs(dim=dim, name=self.name)

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:self.indim] += sigmoid(inbuf)
        outbuf[self.indim:] += 1 - sigmoid(inbuf)

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr += sigmoidPrime(inbuf) * outerr[:self.indim]
        inerr -= sigmoidPrime(inbuf) * outerr[self.indim:]


