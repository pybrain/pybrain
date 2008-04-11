__author__ = 'Daan Wierstra and Tom Schaul'

from scipy import zeros, tanh

from neuronlayer import NeuronLayer
from module import Module
from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.tools.functions import sigmoid, sigmoidPrime, tanhPrime


class LSTMLayer(NeuronLayer):
    """ long short-term memory cell layer """

    sequential = True
    peepholes = False

    def __init__(self, dim, peepholes = False, name = None):
        self.setArgs(dim = dim, peepholes = peepholes)
        # internal buffers:
        self.ingate = zeros((0,dim))
        self.outgate = zeros((0,dim))
        self.forgetgate = zeros((0,dim))
        self.ingatex = zeros((0,dim))
        self.outgatex = zeros((0,dim))
        self.forgetgatex = zeros((0,dim))
        self.state = zeros((0,dim))
        self.ingateError = zeros((0,dim))
        self.outgateError = zeros((0,dim))
        self.forgetgateError = zeros((0,dim))
        self.stateError = zeros((0,dim))
        
        Module.__init__(self, 4*dim, dim, name)
        if self.peepholes:
            self.initParams(dim*3)
            
        # transfer functions and their derivatives
        self.f = sigmoid
        self.fprime = sigmoidPrime
        self.g = lambda x: 2*tanh(x)
        self.gprime = lambda x: 2*tanhPrime(x)
        self.h = self.g
        self.hprime = self.gprime
        
    def _setParameters(self, p, owner = None):
        ParameterContainer._setParameters(self, p, owner)
        dim = self.outdim
        self.ingatePeepWeights = self.params[:dim]
        self.forgetgatePeepWeights = self.params[dim:dim*2]
        self.outgatePeepWeights = self.params[dim*2:]
        
    def _setDerivatives(self, d, owner = None):
        ParameterContainer._setDerivatives(self, d, owner)
        dim = self.outdim
        self.ingatePeepDerivs = self.derivs[:dim]
        self.forgetgatePeepDerivs = self.derivs[dim:dim*2]
        self.outgatePeepDerivs = self.derivs[dim*2:]
        
    def initParams(self, dim, stdParams = 1.):
        ParameterContainer.initParams(self, dim, stdParams)
        self._setParameters(self.params)
        self._setDerivatives(self.derivs)
                        
    def _growBuffers(self):
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
        """ reset buffers to a length (in time dimension) of 1 """
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
        dim = self.outdim
        # slicing the input buffer into the 4 parts
        self.ingatex[self.time] = inbuf[:dim]
        self.forgetgatex[self.time] = inbuf[dim:dim*2]
        cellx = inbuf[dim*2:dim*3]
        self.outgatex[self.time] = inbuf[dim*3:]
        
        # peephole treatment
        if self.peepholes and self.time > 0:
            self.ingatex[self.time] += self.ingatePeepWeights * self.state[self.time-1]
            self.forgetgatex[self.time] += self.forgetgatePeepWeights * self.state[self.time-1]
            
        self.ingate[self.time] = self.f(self.ingatex[self.time])
        self.forgetgate[self.time] = self.f(self.forgetgatex[self.time])
        
        self.state[self.time] = self.ingate[self.time] * self.g(cellx)
        if self.time > 0:
            self.state[self.time] += self.forgetgate[self.time] * self.state[self.time-1]
        
        if self.peepholes:
            self.outgatex[self.time] += self.outgatePeepWeights * self.state[self.time]
        self.outgate[self.time] = self.f(self.outgatex[self.time])
        
        outbuf[:] = self.outgate[self.time] * self.h(self.state[self.time])
    
    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        dim = self.outdim
        cellx = inbuf[dim*2:dim*3]
        
        self.outgateError[self.time] = self.fprime(self.outgatex[self.time]) * outerr * self.h(self.state[self.time])
        self.stateError[self.time] = outerr * self.outgate[self.time] * self.hprime(self.state[self.time])
        if not self._isLastTimestep():
            self.stateError[self.time] += self.stateError[self.time+1] * self.forgetgate[self.time+1]
            if self.peepholes:
                self.stateError[self.time] += self.ingateError[self.time+1] * self.ingatePeepWeights
                self.stateError[self.time] += self.forgetgateError[self.time+1] * self.forgetgatePeepWeights
        if self.peepholes:
            self.stateError[self.time] += self.outgateError[self.time] * self.outgatePeepWeights
        cellError = self.ingate[self.time] * self.gprime(cellx) * self.stateError[self.time]
        if self.time > 0:
            self.forgetgateError[self.time] = self.fprime(self.forgetgatex[self.time]) * self.stateError[self.time] * self.state[self.time-1]
        
        self.ingateError[self.time] = self.fprime(self.ingatex[self.time]) * self.stateError[self.time] * self.g(cellx)    
        
        # compute derivatives
        if self.peepholes:
            self.outgatePeepDerivs += self.outgateError[self.time] * self.state[self.time]
            if self.time > 0:
                self.ingatePeepDerivs += self.ingateError[self.time] * self.state[self.time-1]
                self.forgetgatePeepDerivs += self.forgetgateError[self.time] * self.state[self.time-1]
        
        inerr[:dim] = self.ingateError[self.time]
        inerr[dim:dim*2] = self.forgetgateError[self.time]
        inerr[dim*2:dim*3] = cellError
        inerr[dim*3:] = self.outgateError[self.time]
        
    def whichNeuron(self, inputIndex = None, outputIndex = None):
        if inputIndex != None:
            return inputIndex % self.dim
        if outputIndex != None:
            return outputIndex
        