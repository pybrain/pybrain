# $Id$
__author__ = 'Daan Wierstra and Tom Schaul'

from scipy import zeros

from pybrain.utilities import Named, abstractMethod, substitute
from pybrain.structure.parametercontainer import ParameterContainer


class Module(ParameterContainer, Named):
    """ A module has an input and an output buffer and does some processing 
        to produce the output from the input -- the "forward" method.
        Optionally it can have a "backward" method too, which processes a given output error 
        to derive the input error. 
        
        Input, output and errors are (flat) scipy arrays.
        
        @note: A module memorizes the buffers for all input-output pairs it encounters until reset() is called
    """
    
    # a flag that marks modules that treat a sequence of samples NOT as independent
    sequential = False
    
    def __init__(self, indim, outdim, name = None):
        """ @param indim: the input dimension 
            @param outdim: the output dimension
        """
        self.name = name
        self.time = 0
        self.seqlen = 0
        self.indim = indim
        self.outdim = outdim
        # those buffers are 2D arrays (time, dim)
        self.inputbuffer = zeros((0,self.indim))
        self.outputbuffer = zeros((0,self.outdim))
        self.outputerror = zeros((0,self.outdim))
        self.inputerror = zeros((0,self.indim))
        self._growBuffers()
        
    def _growBuffers(self):
        """ increase the buffer sizes. """
        self.inputbuffer = self._resizeArray(self.inputbuffer)
        self.inputerror = self._resizeArray(self.inputerror)
        self.outputbuffer = self._resizeArray(self.outputbuffer)
        self.outputerror = self._resizeArray(self.outputerror)
        
    def _resetBuffers(self):
        """ reset buffers to a length (in time dimension) of 1 """
        self.inputbuffer = zeros((1,self.indim))
        self.outputbuffer = zeros((1,self.outdim))
        self.outputerror = zeros((1,self.outdim))
        self.inputerror = zeros((1,self.indim))
        
    def _resizeArray(self, a):
        """ increase the buffer size. It should always be one longer than the current sequence length
            and double on every growth step.
        """
        dim = a.shape[1]
        tmp = zeros(((self.seqlen+1)*2, dim))
        tmp[0:self.seqlen] = a
        return tmp
        
    def forward(self, time = None):
        """ produce the output from the input at the specified timestep
            @param time: timestep (default: self-incrementing counter). """
        if time != None:
            self.time = time
        self._forwardImplementation(self.inputbuffer[self.time], self.outputbuffer[self.time])
        self.time += 1
        if self.time > self.seqlen:
            self.seqlen = self.time
        if self.time >= self.outputbuffer.shape[0]:
            self._growBuffers()
        
    def backward(self, time = None):
        """ produce the input error from the output error at the specified timestep
            @param time: timestep (default: self-decrementing counter). """
        if time != None:
            self.time = time
        else:
            self.time -= 1
        self._backwardImplementation(self.outputerror[self.time], self.inputerror[self.time], 
                                     self.outputbuffer[self.time], self.inputbuffer[self.time])        
        
    @substitute('pybrain.tools.pyrex._module.Modulereset')
    def reset(self):
        """ set all the buffers, past and present, to zero. """
        self.time = 0
        self.seqlen = 0
        self.inputbuffer *= 0
        self.outputbuffer *= 0
        self.inputerror *= 0
        self.outputerror *= 0
        
    def _isLastTimestep(self):
        """ used in the recurrent case. """
        # CHECKME
        return self.time == self.seqlen-1
    
    def activateOnDataset(self, dataset ):
        """ Unconditionally runs the module's forward pass on the entire dataset and returns the output. 
        @param dataset: data set to use (FIXME: only tested with SupervisedDataSet!) """        
        dataset.reset()
        self.reset()
        out = zeros( (len(dataset), self.outdim) )
        for i, sample in enumerate(dataset):
            # FIXME: Can we always assume that sample[0] is the input data?
            out[i,:] = self.activate(sample[0])
        self.reset()
        dataset.reset()
        return out
        
    def activate(self, input = None, time = None):
        """ do one transformation of an input, and return the result
            @param input: the input to be transformed (default: what is in the input buffer at the current timestep)
            @param time: timestep (default: self-incrementing counter) """
        if time != None:
            self.time = time
        if input != None:
            self.inputbuffer[self.time] = input
        self.forward()
        return self.outputbuffer[self.time-1]
    
    def backActivate(self, outerr = None, time = None):
        """ do one transformation of an output error backward, and return the result on the input
            @param outerr: the output error to be transformed (default: what is in the buffer at the current timestep)
            @param time: timestep (default: self-decrementing counter) """
        if time != None:
            self.time = time
        if outerr != None:
            self.outputerror[self.time-1] = outerr
        self.backward()
        return self.inputerror[self.time]
        
    def _forwardImplementation(self, inbuf, outbuf):
        """ the actual transformation function of the module """
        abstractMethod()
    
    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        """ the converse of the module's transformation function. 
            If applicable, this should also compute the derivatives of the parameters.
            @note: this method doesn't need to be implemented. """

    