from pybrain.structure.modules.linearlayer import LinearLayer
from pybrain.structure.moduleslice import ModuleSlice
from pybrain.structure.connections.identity import IdentityConnection
from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.structure.connections.shared import MotherConnection, SharedFullConnection
from pybrain.structure.modules.biasunit import BiasUnit
from pybrain.utilities import crossproduct
from pybrain.structure.networks.convolutional import SimpleConvolutionalNetwork

__author__ = 'Tom Schaul, tom@idsia.ch'


class ConvolutionalBoardNetwork(SimpleConvolutionalNetwork):
    """ A type of convolutional network, designed for handling game boards.
    It pads the borders with a uniform bias input to allow one output per board position.
    """

    def __init__(self, boardSize, convSize, numFeatureMaps, **args):
        inputdim = 2
        FeedForwardNetwork.__init__(self, **args)
        inlayer = LinearLayer(inputdim*boardSize*boardSize, name = 'in')
        self.addInputModule(inlayer)

        # we need some treatment of the border too - thus we pad the direct board input.
        x = convSize/2
        insize = boardSize+2*x
        if convSize % 2 == 0:
            insize -= 1
        paddedlayer = LinearLayer(inputdim*insize*insize, name = 'pad')
        self.addModule(paddedlayer)

        # we connect a bias to the padded-parts (with shared but trainable weights).
        bias = BiasUnit()
        self.addModule(bias)
        biasConn = MotherConnection(inputdim)

        paddable = []
        if convSize % 2 == 0:
            xs = range(x)+range(insize-x+1, insize)
        else:
            xs = range(x)+range(insize-x, insize)
        paddable.extend(crossproduct([range(insize), xs]))
        paddable.extend(crossproduct([xs, range(x, boardSize+x)]))

        for (i, j) in paddable:
            self.addConnection(SharedFullConnection(biasConn, bias, paddedlayer,
                                                    outSliceFrom = (i*insize+j)*inputdim,
                                                    outSliceTo = (i*insize+j+1)*inputdim))

        for i in range(boardSize):
            inmod = ModuleSlice(inlayer, outSliceFrom = i*boardSize*inputdim,
                                outSliceTo = (i+1)*boardSize*inputdim)
            outmod = ModuleSlice(paddedlayer, inSliceFrom = ((i+x)*insize+x)*inputdim,
                                 inSliceTo = ((i+x)*insize+x+boardSize)*inputdim)
            self.addConnection(IdentityConnection(inmod, outmod))

        self._buildStructure(inputdim, insize, paddedlayer, convSize, numFeatureMaps)
        self.sortModules()
