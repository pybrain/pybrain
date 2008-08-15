# $Id$
__author__ = 'Daan Wierstra and Tom Schaul'

from scipy import zeros

from pybrain.utilities import abstractMethod, substitute, Named


class Module(Named):
    """A module has an input and an output buffer and does some processing 
    to produce the output from the input -- the "forward" method.
    Optionally it can have a "backward" method too, which processes a given
    output error to derive the input error. 
    
    Input, output and errors are (flat) scipy arrays.
    
    A module memorizes the buffers for all input-output pairs it encounters
    until .reset() is called."""
    
    # Flag that marks modules that treat a sequence of samples not as 
    # independent.
    sequential = False
    
    # Flag which at the same time provides info on how many trainable parameters
    # the module might contain.
    paramdim = 0
        
    def __init__(self, indim, outdim, name=None, **args):
        """Create a Module with an input dimension of indim and an output 
        dimension of outdim."""
        self.setArgs(name=name, **args)
        self.time = 0
        self.seqlen = 0
        self.indim = indim
        self.outdim = outdim
        # Those buffers are 2D arrays (time, dim)
        self._resetBuffers()
        self._growBuffers()
        
    def _growBuffers(self):
        """Increase the buffer sizes."""
        self.inputbuffer = self._resizeArray(self.inputbuffer)
        self.inputerror = self._resizeArray(self.inputerror)
        self.outputbuffer = self._resizeArray(self.outputbuffer)
        self.outputerror = self._resizeArray(self.outputerror)
        
    def _resetBuffers(self):
        """Reset buffers to a length (in time dimension) of 1."""
        self.inputbuffer = zeros((128, self.indim))
        self.outputbuffer = zeros((128, self.outdim))
        self.outputerror = zeros((128, self.outdim))
        self.inputerror = zeros((128, self.indim))
        
    def _resizeArray(self, a):
        """Increase the buffer size. It should always be one longer than the
        current sequence length and double on every growth step."""
        oldsize, dim = a.shape
        seqlen = max(self.seqlen, 1)
        if seqlen < oldsize:
            # No need to grow
            return a
        tmp = zeros((oldsize * 2, dim))
        tmp[0:oldsize] = a
        return tmp
        
    def forward(self, time=None):
        """Produce the output from the input at the specified time.
        
        By default, time is an self incrementing counter."""
        if time is not None:
            self.time = time
        self._forwardImplementation(self.inputbuffer[self.time], 
                                    self.outputbuffer[self.time])
        self.time += 1
        if self.time > self.seqlen:
            self.seqlen = self.time
        if self.time >= self.outputbuffer.shape[0]:
            self._growBuffers()
        
    def backward(self, time = None):
        """Produce the input error from the output error at the specified time.
        
        By default, time is an self incrementing counter."""
        if time is not None:
            self.time = time
        else:
            self.time -= 1
        self._backwardImplementation(self.outputerror[self.time], self.inputerror[self.time], 
                                     self.outputbuffer[self.time], self.inputbuffer[self.time])        
        
    @substitute('pybrain.pyrex._module.Modulereset')
    def reset(self):
        """Set all buffers, past and present, to zero."""
        self.time = 0
        self.seqlen = 0
        self.inputbuffer *= 0
        self.outputbuffer *= 0
        self.inputerror *= 0
        self.outputerror *= 0
        
    def _isLastTimestep(self):
        """Tell wether the current timestep is the last timestep."""
        # CHECKME
        return self.time == self.seqlen-1
    
    def activateOnDataset(self, dataset):
        """Run the module's forward pass on the given dataset unconditionally
        and return the output."""        
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
        """Do one transformation of an input and return the result at the given
        time.
        
        By default, time is an self incrementing counter."""
        if time is not None:
            self.time = time
        if input is not None:
            self.inputbuffer[self.time] = input
        self.forward()
        return self.outputbuffer[self.time - 1].copy()
    
    def backActivate(self, outerr = None, time = None):
        """Do one transformation of an output error outerr backward and return 
        the result on the input at the given time.

        By default, time is an self incrementing counter."""
        if time is not None:
            self.time = time
        if outerr is not None:
            self.outputerror[self.time - 1] = outerr
        self.backward()
        return self.inputerror[self.time].copy()
        
    def _forwardImplementation(self, inbuf, outbuf):
        """Actual forward transformation function. To be overwritten in 
        subclasses."""
        abstractMethod()
    
    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        """Converse of the module's transformation function. Can be overwritten
        in subclasses, does not have to.
        
        Should also compute the derivatives of the parameters."""