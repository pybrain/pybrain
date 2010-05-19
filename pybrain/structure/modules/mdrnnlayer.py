"""The Mdrnn is a bogus layer that only works with fast networks.

It takes an input which is then treated as a multidimensional sequence. E.G. you
might give it an input of `01011010` and specify that its shape is (3, 3), which
results in a 2-dimensional input:

 010
 111
 010
"""


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'

import operator

from pybrain.structure.modules import MDLSTMLayer, LinearLayer, BiasUnit
from pybrain.structure.modules.module import Module
from pybrain.structure.modules.neuronlayer import NeuronLayer
from pybrain.structure.parametercontainer import ParameterContainer


class MdrnnLayer(NeuronLayer, ParameterContainer):
    """Layer that acts as a Multi-Dimensional Recurrent Neural Network, but can
    be integrated more easily into a network.

    Works only for fast networks."""

    # The parameters can be acces via some shortcuts. These are implemented as
    # properties, since the network containing this layer may change the
    # parameters.

    @property
    def inParams(self):
        return self.params[0:self.num_in_params]

    @property
    def predParams(self):
        offset = self.num_in_params
        rest = self.params[offset:]
        return [rest[(i * self.num_rec_params):(i + 1) * self.num_rec_params]
                for i in xrange(self.timedim)]

    @property
    def outParams(self):
        offset = self.num_in_params + self.num_rec_params
        return self.params[offset:offset + self.num_out_params]

    @property
    def biasParams(self):
        offset = self.num_in_params + self.num_rec_params + self.num_out_params
        return self.params[offset:offset + self.num_bias_params]

    def __init__(self, timedim, shape,
                 hiddendim, outsize, blockshape=None, name=None):
        """Initialize an MdrnnLayer.

        The dimensionality of the sequence - for example 2 for a
        picture or 3 for a video - is given by `timedim`, while the sidelengths
        along each dimension are given by the tuple `shape`.

        The layer will have `hiddendim` hidden units per swiping direction. The
        number of swiping directions is given by 2**timedim, which corresponds
        to one swipe from each corner to its opposing corner and back.

        To indicate how many outputs per timesteps are used, you have to specify
        `outsize`.

        In order to treat blocks of the input and not single voxels, you can
        also specify `blockshape`. For example the layer will then feed (2, 2)
        chunks into the network at each timestep which correspond to the (2, 2)
        rectangles that the input can be split into.
        """
        self.timedim = timedim
        self.shape = shape
        blockshape = tuple([1] * timedim) if blockshape is None else blockshape
        self.blockshape = shape
        self.hiddendim = hiddendim
        self.outsize = outsize
        self.indim = reduce(operator.mul, shape, 1)
        self.blocksize = reduce(operator.mul, blockshape, 1)
        self.sequenceLength = self.indim / self.blocksize
        self.outdim = self.sequenceLength * self.outsize

        self.bufferlist = [('cellStates', self.sequenceLength * self.hiddendim)]

        Module.__init__(self, self.indim, self.outdim, name=name)

        # Amount of parameters that are required for the input to the hidden
        self.num_in_params = self.blocksize * self.hiddendim * (3 + self.timedim)

        # Amount of parameters that are needed for the recurrent connections.
        # There is one of the parameter for every time dimension.
        self.num_rec_params = outsize * hiddendim * (3 + self.timedim)

        # Amount of parameters that are needed for the output.
        self.num_out_params = outsize * hiddendim

        # Amount of parameters that are needed from the bias to the hidden and
        # the output
        self.num_bias_params = (3 + self.timedim) * self.hiddendim + self.outsize

        # Total list of parameters.
        self.num_params = sum((self.num_in_params,
                               self.timedim * self.num_rec_params,
                               self.num_out_params,
                               self.num_bias_params))

        ParameterContainer.__init__(self, self.num_params)

        # Some layers for internal use.
        self.hiddenlayer = MDLSTMLayer(self.hiddendim, self.timedim)

        # Every point in the sequence has timedim predecessors.
        self.predlayers = [LinearLayer(self.outsize) for _ in xrange(timedim)]

        # We need a single layer to hold the input. We will swipe a connection
        # over the corrects part of it, in order to feed the correct input in.
        self.inlayer = LinearLayer(self.indim)
        # Make some layers the same to save memory.
        self.inlayer.inputbuffer = self.inlayer.outputbuffer = self.inputbuffer

        # In order to allocate not too much memory, we just set the size of the
        # layer to 1 and correct it afterwards.
        self.outlayer = LinearLayer(self.outdim)
        self.outlayer.inputbuffer = self.outlayer.outputbuffer = self.outputbuffer

        self.bias = BiasUnit()

    def _forwardImplementation(self, inbuf, outbuf):
        raise NotImplementedError("Only for fast networks.")

    def _growBuffers(self):
        super(MdrnnLayer, self)._growBuffers()
        self.inlayer.inputbuffer = self.inlayer.outputbuffer = self.inputbuffer
        self.outlayer.inputbuffer = self.outlayer.outputbuffer = self.outputbuffer

    def _resetBuffers(self, length=1):
        super(MdrnnLayer, self)._resetBuffers()
        if getattr(self, 'inlayer', None) is not None:
            # Don't do this if the buffers have not been set before.
            self.inlayer.inputbuffer = self.inlayer.outputbuffer = self.inputbuffer
            self.outlayer.inputbuffer = self.outlayer.outputbuffer = self.outputbuffer
