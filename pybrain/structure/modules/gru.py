# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:47:02 2015

@author: jackrussell
"""
__original__author__ = 'Daan Wierstra and Tom Schaul'

from scipy import tanh, dot, reshape, outer, zeros

from pybrain.structure.modules.neuronlayer import NeuronLayer
from pybrain.structure.modules.module import Module
from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.tools.functions import sigmoid, sigmoidPrime, tanhPrime


class GRULayer(NeuronLayer, ParameterContainer):
    """Gated Recurrent Unit (GRU) layer.
    
    As proposed in Cho, Kyunghyun, et al: "Learning phrase representations
    using rnn encoder-decoder for statistical machine translation."
    arXiv preprint arXiv:1406.1078 (2014).
    
    The input consists of 3 parts, in the following order:
    - reset gate (r)
    - update gate (z)
    - candidate hidden vector (~h)
    
    Recurrent connections to the gates and candidate "state" are implemented
    by "peep" weights stored in the GRU ParameterContainer, which multiply the
    output from the previous timestep ("prevout")
    
    Therefore this layer should normally NOT be connected to itself
    recursively, as the peep weights take care of all recursion.
    
    """

    sequential = True
    maxoffset = 0

    # Transfer functions and their derivatives
    f = lambda _, x: sigmoid(x)
    fprime = lambda _, x: sigmoidPrime(x)
    g = lambda _, x: tanh(x)
    gprime = lambda _, x: tanhPrime(x)


    def __init__(self, dim, name = None):
        """
        :arg dim: number of cells"""
        self.setArgs(dim = dim)

        # Internal buffers, created dynamically:
        self.bufferlist = [
            ('resetgate', dim),
            ('updategate', dim),
            ('candidate', dim),
            ('resetgatex', dim),
            ('updategatex', dim),
            ('candidatex', dim),
            ('outError', dim),
            ('resetgateError', dim),
            ('updategateError', dim),
            ('candidateError', dim)
        ]

        Module.__init__(self, 3*dim, dim, name)
        ParameterContainer.__init__(self, dim*dim*3)
        self._setParameters(self.params)
        self._setDerivatives(self.derivs)


    def _setParameters(self, p, owner = None):
        ParameterContainer._setParameters(self, p, owner)
        dim = self.outdim ** 2
        self.resetPeepWeights = self.params[:dim]
        self.updatePeepWeights = self.params[dim:dim*2]
        self.candidatePeepWeights = self.params[dim*2:]

    def _setDerivatives(self, d, owner = None):
        ParameterContainer._setDerivatives(self, d, owner)
        dim = self.outdim ** 2
        self.resetPeepDerivs = self.derivs[:dim]
        self.updatePeepDerivs = self.derivs[dim:dim*2]
        self.candidatePeepDerivs = self.derivs[dim*2:]


    def _isLastTimestep(self):
        """Tell whether the current offset is the maximum offset."""
        return self.maxoffset == self.offset

    def _forwardImplementation(self, inbuf, outbuf):
        self.maxoffset = max(self.offset + 1, self.maxoffset)

        dim = self.outdim
        # slicing the input buffer into the 3 parts
        try:
            self.resetgatex[self.offset] = inbuf[:dim]
        except IndexError:
            raise str((self.offset, self.resetgatex.shape))

        self.updategatex[self.offset] = inbuf[dim:dim*2]
        self.candidatex[self.offset] = inbuf[dim*2:]
        
        if self.offset > 0:
            prevout = self.outputbuffer[self.offset-1]

        # peephole treatment
        if self.offset > 0:
            self.resetgatex[self.offset] += dot(reshape(self.resetPeepWeights, (dim, dim)), prevout)
            self.updategatex[self.offset] += dot(reshape(self.updatePeepWeights, (dim, dim)), prevout)

        self.resetgate[self.offset] = self.f(self.resetgatex[self.offset])
        self.updategate[self.offset] = self.f(self.updategatex[self.offset])
        
        if self.offset > 0:
            self.candidatex[self.offset] += self.resetgate[self.offset] * \
            dot(reshape(self.candidatePeepWeights, (dim, dim)), prevout)
            
        self.candidate[self.offset] = self.g(self.candidatex[self.offset])

        outbuf[:] = (1 - self.updategate[self.offset]) * self.candidate[self.offset]
        if self.offset > 0:
            outbuf[:] += self.updategate[self.offset] * prevout
            

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        dim = self.outdim
        z = self.updategate[self.offset]
        r = self.resetgate[self.offset]
        
        if self.offset > 0:
            prevout = self.outputbuffer[self.offset-1]
        else:
            prevout = zeros(dim)

        # Output error from current timestep
        self.outError[self.offset] = outerr
        
        # Output errors backpropagated in time from next timestep
        if not self._isLastTimestep():
            self.contrib1[self.offset] = self.updategate[self.offset+1] * self.outError[self.offset+1]
            self.contrib2[self.offset] = dot(reshape(self.resetPeepWeights, (dim, dim)).T, self.resetgateError[self.offset+1]) 
            self.contrib3[self.offset] = dot(reshape(self.updatePeepWeights, (dim, dim)).T, self.updategateError[self.offset+1])
            self.contrib4[self.offset] = dot(reshape(self.candidatePeepWeights, (dim, dim)).T,
                                            self.resetgate[self.offset+1] * self.candidateError[self.offset+1])
            
            self.outError[self.offset] += self.updategate[self.offset+1] * self.outError[self.offset+1]
            self.outError[self.offset] += dot(reshape(self.resetPeepWeights, (dim, dim)).T, self.resetgateError[self.offset+1]) 
            self.outError[self.offset] += dot(reshape(self.updatePeepWeights, (dim, dim)).T, self.updategateError[self.offset+1])
            self.outError[self.offset] += dot(reshape(self.candidatePeepWeights, (dim, dim)).T,
                                            self.resetgate[self.offset+1] * self.candidateError[self.offset+1])
        
        self.candidateError[self.offset] = (1 - z) * self.gprime(self.candidatex[self.offset]) \
                                            * self.outError[self.offset]
        
        self.updategateError[self.offset] = self.fprime(self.updategatex[self.offset]) \
                                        * (prevout - self.candidate[self.offset]) \
                                        * self.outError[self.offset]                                            
        self.resetgateError[self.offset] = self.fprime(self.resetgatex[self.offset]) \
                                        * dot(reshape(self.candidatePeepWeights, (dim, dim)), prevout) \
                                        * self.candidateError[self.offset]

        # compute peep derivatives
        if self.offset > 0:            
            self.resetPeepDerivs += outer(self.resetgateError[self.offset], prevout).flatten()
            self.updatePeepDerivs += outer(self.updategateError[self.offset], prevout).flatten()
            self.candidatePeepDerivs += outer(self.candidateError[self.offset] * r, prevout).flatten()

        # compute out errors
        inerr[:dim] = self.resetgateError[self.offset]
        inerr[dim:dim*2] = self.updategateError[self.offset]
        inerr[dim*2:] = self.candidateError[self.offset]
        

    def whichNeuron(self, inputIndex = None, outputIndex = None):
        if inputIndex != None:
            return inputIndex % self.dim
        if outputIndex != None:
            return outputIndex
