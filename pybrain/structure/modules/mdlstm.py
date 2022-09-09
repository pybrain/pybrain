__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import zeros, tanh

from pybrain.structure.modules.neuronlayer import NeuronLayer
from pybrain.structure.modules.module import Module
from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.tools.functions import sigmoid, sigmoidPrime, tanhPrime
from pybrain.structure.moduleslice import ModuleSlice


class MDLSTMLayer(NeuronLayer, ParameterContainer):
    """Multi-dimensional long short-term memory cell layer.

    The cell-states are explicitly passed on through a part of
    the input/output buffers (which should be connected correctly with IdentityConnections).

    The input consists of 4 parts, in the following order:
    - input gate
    - forget gates (1 per dim)
    - cell input
    - output gate
    - previous states (1 per dim)

    The output consists of two parts:
    - cell output
    - current statte


    Attention: this module has to be used with care: it's last <size> input and
    outputs are reserved for transmitting internal states on flattened recursive
    multi-dim networks, and so its connections have always to be sliced!
    """

    peepholes = False
    dimensions = 1
    maxoffset = 0

    # Transfer functions and their derivatives
    def f(self, x): return sigmoid(x)
    def fprime(self, x): return sigmoidPrime(x)
    def g(self, x): return tanh(x)
    def gprime(self, x): return tanhPrime(x)
    def h(self, x): return tanh(x)
    def hprime(self, x): return tanhPrime(x)

    def __init__(self, dim, dimensions=1, peepholes=False, name=None):
        self.setArgs(dim=dim, peepholes=peepholes, dimensions=dimensions)

        # Internal buffers:
        self.bufferlist = [
            ('ingate', dim),
            ('outgate', dim),
            ('forgetgate', dim * dimensions),
            ('ingatex', dim),
            ('outgatex', dim),
            ('forgetgatex', dim * dimensions),
            ('state', dim),
            ('ingateError', dim),
            ('outgateError', dim),
            ('forgetgateError', dim * dimensions),
            ('stateError', dim),
        ]

        Module.__init__(self, (3 + 2 * dimensions) * dim, dim * 2, name)

        if self.peepholes:
            ParameterContainer.__init__(self, dim * (2 + dimensions))
            self._setParameters(self.params)
            self._setDerivatives(self.derivs)

    def _setParameters(self, p, owner=None):
        ParameterContainer._setParameters(self, p, owner)
        size = self.dim
        self.ingatePeepWeights = self.params[:size]
        self.forgetgatePeepWeights = self.params[size:size*(1 + self.dimensions)]
        self.outgatePeepWeights = self.params[size*(1 + self.dimensions):]

    def _setDerivatives(self, d, owner=None):
        ParameterContainer._setDerivatives(self, d, owner)
        size = self.dim
        self.ingatePeepDerivs = self.derivs[:size]
        self.forgetgatePeepDerivs = \
            self.derivs[size:size * (1 + self.dimensions)]
        self.outgatePeepDerivs = \
            self.derivs[size * (1 + self.dimensions):]

    def _forwardImplementation(self, inbuf, outbuf):
        self.maxoffset = max(self.offset + 1, self.maxoffset)
        size = self.dim
        # slicing the input buffer into the 4 parts.
        self.ingatex[self.offset] = inbuf[:size]
        self.forgetgatex[self.offset] = inbuf[size:size*(1+self.dimensions)]
        cellx = inbuf[size*(1+self.dimensions):size*(2+self.dimensions)]
        self.outgatex[self.offset] = inbuf[size*(2+self.dimensions):size*(3+self.dimensions)]
        laststates = inbuf[size*(3+self.dimensions):]

        # Peephole treatment
        if self.peepholes:
            for i in range(self.dimensions):
                self.ingatex[self.offset] += self.ingatePeepWeights * laststates[size * i:size * (i + 1)]
            self.forgetgatex[self.offset] += self.forgetgatePeepWeights * laststates

        self.ingate[self.offset] = self.f(self.ingatex[self.offset])
        self.forgetgate[self.offset] = self.f(self.forgetgatex[self.offset])

        self.state[self.offset] = self.ingate[self.offset] * self.g(cellx)
        for i in range(self.dimensions):
            self.state[self.offset] += self.forgetgate[self.offset, size*i:size*(i+1)] * laststates[size*i:size*(i+1)]

        if self.peepholes:
            self.outgatex[self.offset] += self.outgatePeepWeights * self.state[self.offset]
        self.outgate[self.offset] = self.f(self.outgatex[self.offset])

        outbuf[:size] = self.outgate[self.offset] * self.h(self.state[self.offset])
        outbuf[size:] = self.state[self.offset]

    def _backwardImplementation(self, outerr2, inerr, outbuf, inbuf):
        size = self.dim
        cellx = inbuf[size*(1+self.dimensions):size*(2+self.dimensions)]
        laststates = inbuf[size*(3+self.dimensions):]
        outerr = outerr2[:size]
        nextstateerr = outerr2[size:]

        self.outgateError[self.offset] = self.fprime(self.outgatex[self.offset]) * outerr * self.h(self.state[self.offset])
        self.stateError[self.offset] = outerr * self.outgate[self.offset] * self.hprime(self.state[self.offset])
        self.stateError[self.offset] += nextstateerr
        if self.peepholes:
            self.stateError[self.offset] += self.outgateError[self.offset] * self.outgatePeepWeights
        cellError = self.ingate[self.offset] * self.gprime(cellx) * self.stateError[self.offset]
        for i in range(self.dimensions):
            self.forgetgateError[self.offset, size*i:size*(i+1)] = (self.fprime(self.forgetgatex[self.offset, size*i:size*(i+1)])
                                                                  * self.stateError[self.offset] * laststates[size*i:size*(i+1)])

        self.ingateError[self.offset] = self.fprime(self.ingatex[self.offset]) * self.stateError[self.offset] * self.g(cellx)

        # compute derivatives
        if self.peepholes:
            self.outgatePeepDerivs += self.outgateError[self.offset] * self.state[self.offset]
            for i in range(self.dimensions):
                self.ingatePeepDerivs += self.ingateError[self.offset] * laststates[size*i:size*(i+1)]
                self.forgetgatePeepDerivs[size*i:size*(i+1)] += (self.forgetgateError[self.offset, size*i:size*(i+1)]
                                                                 * laststates[size*i:size*(i+1)])

        instateErrors = zeros((size * self.dimensions))
        for i in range(self.dimensions):
            instateErrors[size * i:size * (i + 1)] = (self.stateError[self.offset] *
                                                      self.forgetgate[self.offset, size*i:size*(i+1)])
            if self.peepholes:
                instateErrors[size * i:size * (i + 1)] += self.ingateError[self.offset] * self.ingatePeepWeights
                instateErrors[size * i:size * (i + 1)] += self.forgetgateError[self.offset, size*i:size*(i+1)] * \
                                                          self.forgetgatePeepWeights[size*i:size*(i+1)]

        inerr[:size] = self.ingateError[self.offset]
        inerr[size:size*(1+self.dimensions)] = self.forgetgateError[self.offset]
        inerr[size*(1+self.dimensions):size*(2+self.dimensions)] = cellError
        inerr[size*(2+self.dimensions):size*(3+self.dimensions)] = self.outgateError[self.offset]
        inerr[size * (3 + self.dimensions):] = instateErrors

    def meatSlice(self):
        """Return a moduleslice that wraps the meat part of the layer."""
        return ModuleSlice(self,
                           inSliceTo=self.dim * (3 + self.dimensions),
                           outSliceTo=self.dim)

    def stateSlice(self):
        """Return a moduleslice that wraps the state transfer part of the layer.
        """
        return ModuleSlice(self,
                           inSliceFrom=self.dim * (3 + self.dimensions),
                           outSliceFrom=self.dim)

    def whichNeuron(self, inputIndex=None, outputIndex=None):
        if inputIndex != None:
            return inputIndex % self.dim
        if outputIndex != None:
            return outputIndex % self.dim