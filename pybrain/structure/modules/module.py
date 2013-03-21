__author__ = 'Daan Wierstra and Tom Schaul'

from scipy import append, zeros

from pybrain.utilities import abstractMethod, Named


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

    # An offset that is added upon any array access. Useful for implementing
    # things like time.
    offset = 0

    bufferlist = None

    def __init__(self, indim, outdim, name=None, **args):
        """Create a Module with an input dimension of indim and an output
        dimension of outdim."""
        self.setArgs(name=name, **args)

        # Make sure that it does not matter whether Module.__init__ is called
        # before or after adding elements to bufferlist in subclasses.
        # TODO: it should be possible to use less than these buffers. For some
        # methods, an error is not completely necessary. (e.g. evolution)
        self.bufferlist = [] if not self.bufferlist else self.bufferlist
        self.bufferlist += [('inputbuffer', indim),
                            ('inputerror', indim),
                            ('outputbuffer', outdim),
                            ('outputerror', outdim), ]

        self.indim = indim
        self.outdim = outdim
        # Those buffers are 2D arrays (time, dim)
        self._resetBuffers()

    def _resetBuffers(self, length=1):
        """Reset buffers to a length (in time dimension) of 1."""
        for buffername, dim in self.bufferlist:
            setattr(self, buffername, zeros((length, dim)))
        if length==1:
            self.offset = 0

    def _growBuffers(self):
        """Double the size of the modules buffers in its first dimension and
        keep the current values."""
        currentlength = getattr(self, self.bufferlist[0][0]).shape[0]
        # Save the current buffers
        tmp = [getattr(self, n) for n, _ in self.bufferlist]
        Module._resetBuffers(self, currentlength * 2)

        for previous, (buffername, _dim) in zip(tmp, self.bufferlist):
            buffer_ = getattr(self, buffername)
            buffer_[:currentlength] = previous

    def forward(self):
        """Produce the output from the input."""
        self._forwardImplementation(self.inputbuffer[self.offset],
                                    self.outputbuffer[self.offset])

    def backward(self):
        """Produce the input error from the output error."""
        self._backwardImplementation(self.outputerror[self.offset],
                                     self.inputerror[self.offset],
                                     self.outputbuffer[self.offset],
                                     self.inputbuffer[self.offset])

    def reset(self):
        """Set all buffers, past and present, to zero."""
        self.offset = 0
        for buffername, l  in self.bufferlist:
            buf = getattr(self, buffername)
            buf[:] = zeros(l)

    def shift(self, items):
        """Shift all buffers up or down a defined number of items on offset axis.
        Negative values indicate backward shift."""
        if items == 0:
            return
        self.offset += items
        for buffername, _  in self.bufferlist:
            buf = getattr(self, buffername)
            assert abs(items) <= len(buf), "Cannot shift further than length of buffer."
            fill = zeros((abs(items), len(buf[0])))
            if items < 0:
                buf[:] = append(buf[-items:], fill, 0)
            else:
                buf[:] = append(fill ,buf[0:-items] , 0)

    def activateOnDataset(self, dataset):
        """Run the module's forward pass on the given dataset unconditionally
        and return the output."""
        dataset.reset()
        self.reset()
        out = zeros((len(dataset), self.outdim))
        for i, sample in enumerate(dataset):
            # FIXME: Can we always assume that sample[0] is the input data?
            out[i, :] = self.activate(sample[0])
        self.reset()
        dataset.reset()
        return out

    def activate(self, inpt):
        """Do one transformation of an input and return the result."""
        assert len(self.inputbuffer[self.offset]) == len(inpt), str((len(self.inputbuffer[self.offset]), len(inpt)))
        self.inputbuffer[self.offset] = inpt
        self.forward()
        return self.outputbuffer[self.offset].copy()

    def backActivate(self, outerr):
        """Do one transformation of an output error outerr backward and return
        the error on the input."""
        self.outputerror[self.offset] = outerr
        self.backward()
        return self.inputerror[self.offset].copy()

    def _forwardImplementation(self, inbuf, outbuf):
        """Actual forward transformation function. To be overwritten in
        subclasses."""
        abstractMethod()

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        """Converse of the module's transformation function. Can be overwritten
        in subclasses, does not have to.

        Should also compute the derivatives of the parameters."""
