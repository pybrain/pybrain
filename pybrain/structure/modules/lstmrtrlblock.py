#@PydevCodeAnalysisIgnore
__author__ = 'Daan Wierstra, daan@idsia.ch'

from scipy import zeros, tanh

from module import Module
from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.tools.functions import sigmoid, sigmoidPrime, tanhPrime


class LSTMRTRLBlock(Module, ParameterContainer):
    """ long short-term memory implemented with RTRL; incoming connections
        and recurrent connections are included within this block! """

    sequential = True

    def __init__(self, indim, outdim, peepholes = False, name = None):
        nrNeurons = outdim
        self.peep = peepholes
        # internal buffers:
        self.ingate = zeros((0,nrNeurons))
        self.outgate = zeros((0,nrNeurons))
        self.forgetgate = zeros((0,nrNeurons))
        self.cell = zeros((0,nrNeurons))
        self.ingatex = zeros((0,nrNeurons))
        self.outgatex = zeros((0,nrNeurons))
        self.forgetgatex = zeros((0,nrNeurons))
        self.cellx = zeros((0,nrNeurons))
        self.state = zeros((0,nrNeurons))
        self.ingateError = zeros((0,nrNeurons))
        self.outgateError = zeros((0,nrNeurons))
        self.forgetgateError = zeros((0,nrNeurons))
        self.stateError = zeros((0,nrNeurons))
        self.Sin = zeros((0,indim*nrNeurons))
        self.Sforget = zeros((0,indim*nrNeurons))
        self.Scell = zeros((0,indim*nrNeurons))
        self.SinRec = zeros((0,indim*nrNeurons))
        self.SforgetRec = zeros((0,indim*nrNeurons))
        self.ScellRec = zeros((0,indim*nrNeurons))
        
        Module.__init__(self, indim, outdim, name)
        if self.peep:
            ParameterContainer.__init__(self, nrNeurons*3 + 8*indim*nrNeurons)
            self.Sin_peep = zeros((0,nrNeurons))
            self.Sforget_peep = zeros((0,nrNeurons))
            self.Scell_peep = zeros((0,nrNeurons))
        else:
            ParameterContainer.__init__(self, 8*dim*nrNeurons)
        self._setParameters(self.params)
        self._setDerivatives(self.derivs)
            
        # transfer functions and their derivatives
        self.f = sigmoid
        self.fprime = sigmoidPrime
        self.g = lambda x: 2*tanh(x)
        self.gprime = lambda x: 2*tanhPrime(x)
        self.h = self.g
        self.hprime = self.gprime
        
    def _setParameters(self, p, owner = None):
        ParameterContainer._setParameters(self, p, owner)
        nrNeurons = self.outdim
        self.ingateConns = self.params[:indim*nrNeurons]
        self.forgetgateConns = self.params[indim*nrNeurons:2*indim*nrNeurons]
        self.cellConns = self.params[2*indim*nrNeurons:3*indim*nrNeurons]
        self.outgateConns = self.params[3*indim*nrNeurons:4*indim*nrNeurons]
        self.ingateRecConns = self.params[4*indim*nrNeurons:5*indim*nrNeurons]
        self.forgetgateRecConns = self.params[5*indim*nrNeurons:6*indim*nrNeurons]
        self.cellRecConns = self.params[6*indim*nrNeurons:7*indim*nrNeurons]
        self.outgateRecConns = self.params[7*indim*nrNeurons:8*indim*nrNeurons]
        if self.peep:
            self.ingatePeepWeights = self.params[8*indim*nrNeurons:8*indim*nrNeurons+nrNeurons]
            self.forgetgatePeepWeights = self.params[8*indim*nrNeurons+nrNeurons:8*indim*nrNeurons+nrNeurons*2]
            self.outgatePeepWeights = self.params[8*indim*nrNeurons+nrNeurons*2:]
        
    def _setDerivatives(self, d, owner = None):
        ParameterContainer._setDerivatives(self, d, owner)
        nrNeurons = self.outdim
        self.ingateConnDerivs = self.derivs[:indim*nrNeurons]
        self.forgetgateConnDerivs = self.derivs[indim*nrNeurons:2*indim*nrNeurons]
        self.cellConnDerivs = self.derivs[2*indim*nrNeurons:3*indim*nrNeurons]
        self.outgateConnDerivs = self.derivs[3*indim*nrNeurons:4*indim*nrNeurons]
        self.ingateRecConnDerivs = self.derivs[4*indim*nrNeurons:5*indim*nrNeurons]
        self.forgetgateRecConnDerivs = self.derivs[5*indim*nrNeurons:6*indim*nrNeurons]
        self.cellRecConnDerivs = self.derivs[6*indim*nrNeurons:7*indim*nrNeurons]
        self.outgateRecConnDerivs = self.derivs[7*indim*nrNeurons:8*indim*nrNeurons]
        if self.peep:
            self.ingatePeepDerivs = self.derivs[8*indim*nrNeurons:8*indim*nrNeurons+nrNeurons]
            self.forgetgatePeepDerivs = self.derivs[8*indim*nrNeurons+nrNeurons:8*indim*nrNeurons+nrNeurons*2]
            self.outgatePeepDerivs = self.derivs[8*indim*nrNeurons+nrNeurons*2:]        
                        
    def _growBuffers(self):
        Module._growBuffers(self)
        self.ingate = self._resizeArray(self.ingate)
        self.outgate = self._resizeArray(self.outgate)
        self.forgetgate = self._resizeArray(self.forgetgate)
        self.ingatex = self._resizeArray(self.ingatex)
        self.outgatex = self._resizeArray(self.outgatex)
        self.forgetgatex = self._resizeArray(self.forgetgatex)
        self.cellx = self._resizeArray(self.cellx)
        self.state = self._resizeArray(self.state)
        self.ingateError = self._resizeArray(self.ingateError)
        self.outgateError = self._resizeArray(self.outgateError)
        self.forgetgateError = self._resizeArray(self.forgetgateError)
        self.stateError = self._resizeArray(self.stateError)
        self.Sin = self._resizeArray(self.Sin)
        self.Sforget = self._resizeArray(self.Sforget)
        self.Scell = self._resizeArray(self.Scell)
        self.SinRec = self._resizeArray(self.Sin)
        self.SforgetRec = self._resizeArray(self.Sforget)
        self.ScellRec = self._resizeArray(self.Scell)
        if self.peep:
            self.Sin_peep = self._resizeArray(self.Sin_peep)
            self.Sforget_peep = self._resizeArray(self.Sforget_peep)
            self.Scell_peep = self._resizeArray(self.Scell_peep)
        
    def reset(self):
        Module.reset(self)
        self.ingate *= 0
        self.outgate *= 0
        self.forgetgate *= 0
        self.ingatex *= 0
        self.cellx *= 0
        self.outgatex *= 0
        self.forgetgatex *= 0
        self.state *= 0
        self.ingateError *= 0
        self.outgateError *= 0
        self.forgetgateError *= 0
        self.stateError *= 0
        self.Sin *=0
        self.Sforget *= 0
        self.Scell *= 0
        self.SinRec *=0
        self.SforgetRec *=0
        self.ScellRec *=0
        if self.peep:
            self.Sin_peep *=0
            self.Scell_peep *= 0
            self.Sforget_peep *= 0
        
        # todo: set state derivs to 0?
        
    def _forwardImplementation(self, inbuf, outbuf):
        nrNeurons = self.outdim
        # slicing the input buffer into the 4 parts
        self.ingatex[self.time] = dot(reshape(ingateConns, (self.outdim, self.indim)), inbuf)
        if self.time > 0:
            self.ingatex[self.time] += dot(reshape(ingateRecConns, (self.outdim, nrNeurons)), outputbuffer[self.time - 1])
        self.forgetgatex[self.time] = dot(reshape(forgetgateConns, (self.outdim, self.indim)), inbuf)
        if self.time > 0:
            self.forgetgatex[self.time] += dot(reshape(forgetgateRecConns, (self.outdim, nrNeurons)), outputbuffer[self.time - 1])
        self.cellx[self.time] = dot(reshape(cellConns, (self.outdim, self.indim)), inbuf)
        if self.time > 0:
            self.cellx[self.time] += dot(reshape(cellRecConns, (self.outdim, nrNeurons)), outputbuffer[self.time - 1])
        self.outgatex[self.time] = dot(reshape(outgateConns, (self.outdim, self.indim)), inbuf)
        if self.time > 0:
            self.outgatex[self.time] += dot(reshape(outgateRecConns, (self.outdim, nrNeurons)), outputbuffer[self.time - 1])
        
        # peephole treatment
        if self.peep and self.time > 0:
            self.ingatex[self.time] += self.ingatePeepWeights * self.state[self.time-1]
            self.forgetgatex[self.time] += self.forgetgatePeepWeights * self.state[self.time-1]
            
        self.ingate[self.time] = self.f(self.ingatex[self.time])
        self.forgetgate[self.time] = self.f(self.forgetgatex[self.time])
        
        self.state[self.time] = self.ingate[self.time] * self.g(self.cellx[self.time])
        if self.time > 0:
            self.state[self.time] += self.forgetgate[self.time] * self.state[self.time-1]
        
        if self.peep:
            self.outgatex[self.time] += self.outgatePeepWeights * self.state[self.time]
        self.outgate[self.time] = self.f(self.outgatex[self.time])
        
        outbuf[:] = self.outgate[self.time] * self.h(self.state[self.time])
        
        if self.time > 0:
            self.Scell[self.time] = self.Scell[self.time - 1]*self.forgetgate[self.time] + \
                                  self.gprime(self.cellx[self.time]) * self.ingate[self.time] 
        
    
    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        nrNeurons = self.outdim
        cellx = inbuf[nrNeurons*2:nrNeurons*3]
        
        self.outgateError[self.time] = self.fprime(self.outgatex[self.time]) * outerr * self.h(self.state[self.time])
        self.stateError[self.time] = outerr * self.outgate[self.time] * self.hprime(self.state[self.time])
        if not self._isLastTimestep():
            self.stateError[self.time] += self.stateError[self.time+1] * self.forgetgate[self.time+1]
            if self.peep:
                self.stateError[self.time] += self.ingateError[self.time+1] * self.ingatePeepWeights
                self.stateError[self.time] += self.forgetgateError[self.time+1] * self.forgetgatePeepWeights
        if self.peep:
            self.stateError[self.time] += self.outgateError[self.time] * self.outgatePeepWeights
        cellError = self.ingate[self.time] * self.gprime(cellx) * self.stateError[self.time]
        if self.time > 0:
            self.forgetgateError[self.time] = self.fprime(self.forgetgatex[self.time]) * self.stateError[self.time] * self.state[self.time-1]
        
        self.ingateError[self.time] = self.fprime(self.ingatex[self.time]) * self.stateError[self.time] * self.g(cellx)    
        
        # compute derivatives
        if self.peep:
            self.outgatePeepDerivs += self.outgateError[self.time] * self.state[self.time]
            if self.time > 0:
                self.ingatePeepDerivs += self.ingateError[self.time] * self.state[self.time-1]
                self.forgetgatePeepDerivs += self.forgetgateError[self.time] * self.state[self.time-1]
        
        inerr[:nrNeurons] = self.ingateError[self.time]
        inerr[nrNeurons:nrNeurons*2] = self.forgetgateError[self.time]
        inerr[nrNeurons*2:nrNeurons*3] = cellError
        inerr[nrNeurons*3:] = self.outgateError[self.time]
        