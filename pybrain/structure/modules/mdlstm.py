__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import zeros, tanh

from neuronlayer import NeuronLayer
from module import Module
from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.tools.functions import sigmoid, sigmoidPrime, tanhPrime
from pybrain.structure.moduleslice import ModuleSlice


class MDLSTMLayer(NeuronLayer, ParameterContainer):
    """Multi-dimensional long short-term memory cell layer.
    
    Attention: this module has to be used with care: it's last <size> input and
    outputs are reserved for transmitting internal states on flattened recursive
    multi-dim networks, and so int's connections have always to be sliced!""" 
    
    peepholes = False
    dimensions = 1   
    
    def __init__(self, dim, dimensions=1, peepholes=False, name=None):
        self.setArgs(dim=dim, peepholes=peepholes, dimensions=dimensions)
        
        # Internal buffers:
        self.ingate = zeros((0,dim))
        self.outgate = zeros((0,dim))
        self.forgetgate = zeros((0,dim*dimensions))
        self.ingatex = zeros((0,dim))
        self.outgatex = zeros((0,dim))
        self.forgetgatex = zeros((0,dim*dimensions))
        self.state = zeros((0,dim))
        self.ingateError = zeros((0,dim))
        self.outgateError = zeros((0,dim))
        self.forgetgateError = zeros((0,dim*dimensions))
        self.stateError = zeros((0,dim))
        
        Module.__init__(self, (3 + 2 * dimensions) * dim, dim * 2, name)
        
        if self.peepholes:
            ParameterContainer.__init__(self, dim * (2 + dimensions))
            self._setParameters(self.params)
            self._setDerivatives(self.derivs)        
            
        # Transfer functions and their derivatives
        self.f = sigmoid
        self.fprime = sigmoidPrime
        self.g = lambda x: 2 * tanh(x)
        self.gprime = lambda x: 2 * tanhPrime(x)
        self.h = self.g
        self.hprime = self.gprime
        
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
        
    def _growBuffers(self):
        """Increase the buffer sizes."""
        Module._growBuffers(self)
        self.ingate = self._resizeArray(self.ingate)
        self.outgate = self._resizeArray(self.outgate)
        self.forgetgate = self._resizeArray(self.forgetgate)
        self.ingatex = self._resizeArray(self.ingatex)
        self.outgatex = self._resizeArray(self.outgatex)
        self.forgetgatex = self._resizeArray(self.forgetgatex)
        self.state = self._resizeArray(self.state)
        self.ingateError = self._resizeArray(self.ingateError)
        self.outgateError = self._resizeArray(self.outgateError)
        self.forgetgateError = self._resizeArray(self.forgetgateError)
        self.stateError = self._resizeArray(self.stateError)
        
    def _resetBuffers(self):
        """Reset buffers to a length (in time dimension) of 1."""
        Module._resetBuffers(self)
        self.ingate = zeros((1,self.dim))
        self.outgate = zeros((1,self.dim))
        self.forgetgate = zeros((1,self.dim))
        self.ingatex = zeros((1,self.dim))
        self.outgatex = zeros((1,self.dim))
        self.forgetgatex = zeros((1,self.dim))
        self.state = zeros((1,self.dim))
        self.ingateError = zeros((1,self.dim))
        self.outgateError = zeros((1,self.dim))
        self.forgetgateError = zeros((1,self.dim))
        self.stateError = zeros((1,self.dim))
        
    def reset(self):
        Module.reset(self)
        self.ingate *= 0
        self.outgate *= 0
        self.forgetgate *= 0
        self.ingatex *= 0
        self.outgatex *= 0
        self.forgetgatex *= 0
        self.state *= 0
        self.ingateError *= 0
        self.outgateError *= 0
        self.forgetgateError *= 0
        self.stateError *= 0
        
    def _forwardImplementation(self, inbuf, outbuf):
        size = self.dim
        # slicing the input buffer into the 4 parts.
        self.ingatex[self.time] = inbuf[:size]
        self.forgetgatex[self.time] = inbuf[size:size*(1+self.dimensions)]
        cellx = inbuf[size*(1+self.dimensions):size*(2+self.dimensions)]
        self.outgatex[self.time] = inbuf[size*(2+self.dimensions):size*(3+self.dimensions)]        
        laststates = inbuf[size*(3+self.dimensions):]
        
        # Peephole treatment
        if self.peepholes and self.time > 0:
            self.ingatex[self.time] += self.ingatePeepWeights * self.state[self.time-1]
            self.forgetgatex[self.time] += self.forgetgatePeepWeights * laststates
            
        self.ingate[self.time] = self.f(self.ingatex[self.time])
        self.forgetgate[self.time] = self.f(self.forgetgatex[self.time])
        
        self.state[self.time] = self.ingate[self.time] * self.g(cellx)
        if self.time > 0:
            for i in range(self.dimensions):
                self.state[self.time] += self.forgetgate[self.time, size*i:size*(i+1)] * laststates[size*i:size*(i+1)]
        
        if self.peepholes:
            self.outgatex[self.time] += self.outgatePeepWeights * self.state[self.time]
        self.outgate[self.time] = self.f(self.outgatex[self.time])
        
        outbuf[:size] = self.outgate[self.time] * self.h(self.state[self.time])
        outbuf[size:] = self.state[self.time]
    
    def _backwardImplementation(self, outerr2, inerr, outbuf, inbuf):
        size = self.dim
        cellx = inbuf[size*(1+self.dimensions):size*(2+self.dimensions)]
        laststates = inbuf[size*(3+self.dimensions):]
        outerr = outerr2[:size]
        nextstateerr = outerr2[size:]
        
        self.outgateError[self.time] = self.fprime(self.outgatex[self.time]) * outerr * self.h(self.state[self.time])
        self.stateError[self.time] = outerr * self.outgate[self.time] * self.hprime(self.state[self.time])
        if not self._isLastTimestep():
            self.stateError[self.time] += nextstateerr
            if self.peepholes:
                self.stateError[self.time] += self.ingateError[self.time+1] * self.ingatePeepWeights
                for i in range(self.dimensions):
                    self.stateError[self.time] += (self.forgetgateError[self.time+1, size*i:size*(i+1)] * 
                                                   self.forgetgatePeepWeights[size*i:size*(i+1)])
        if self.peepholes:
            self.stateError[self.time] += self.outgateError[self.time] * self.outgatePeepWeights
        cellError = self.ingate[self.time] * self.gprime(cellx) * self.stateError[self.time]
        if self.time > 0:
            for i in range(self.dimensions):
                self.forgetgateError[self.time, size*i:size*(i+1)] = (self.fprime(self.forgetgatex[self.time, size*i:size*(i+1)]) 
                                                                      * self.stateError[self.time] * laststates[size*i:size*(i+1)])
        
        self.ingateError[self.time] = self.fprime(self.ingatex[self.time]) * self.stateError[self.time] * self.g(cellx)    
        
        # compute derivatives
        if self.peepholes:
            self.outgatePeepDerivs += self.outgateError[self.time] * self.state[self.time]
            if self.time > 0:
                self.ingatePeepDerivs += self.ingateError[self.time] * self.state[self.time-1]
                for i in range(self.dimensions):
                    self.forgetgatePeepDerivs[size*i:size*(i+1)] += (self.forgetgateError[self.time, size*i:size*(i+1)] 
                                                                     * laststates[size*i:size*(i+1)])
        
        inerr[:size] = self.ingateError[self.time]
        inerr[size:size*(1+self.dimensions)] = self.forgetgateError[self.time]
        inerr[size*(1+self.dimensions):size*(2+self.dimensions)] = cellError
        inerr[size*(2+self.dimensions):size*(3+self.dimensions)] = self.outgateError[self.time]
        for i in range(self.dimensions):
            inerr[size*(3+self.dimensions+i):size*(3+self.dimensions+i+1)] = (self.stateError[self.time] * 
                                                                  self.forgetgate[self.time, size*i:size*(i+1)])
            
    def meatSlice(self):
        """eturn a moduleslice that wraps the meat part of the layer."""
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