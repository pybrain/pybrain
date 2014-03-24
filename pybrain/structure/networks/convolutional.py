from pybrain.structure.modules.linearlayer import LinearLayer
from pybrain.structure.modules.tanhlayer import TanhLayer
from pybrain.structure.moduleslice import ModuleSlice
from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.structure.connections.shared import MotherConnection, SharedFullConnection
from pybrain.structure.modules.sigmoidlayer import SigmoidLayer

__author__ = 'Tom Schaul, tom@idsia.ch'

# TODO: code up a more general version
# TODO: use modulemash.viewonflatlayer()


class SimpleConvolutionalNetwork(FeedForwardNetwork):
    """ A network with a specific form of weight-sharing, on a single 2D layer,
    convoluting neighboring inputs (within a square). """

    def __init__(self, inputdim, insize, convSize, numFeatureMaps, **args):
        FeedForwardNetwork.__init__(self, **args)
        inlayer = LinearLayer(inputdim * insize * insize)
        self.addInputModule(inlayer)
        self._buildStructure(inputdim, insize, inlayer, convSize, numFeatureMaps)
        self.sortModules()


    def _buildStructure(self, inputdim, insize, inlayer, convSize, numFeatureMaps):
        #build layers
        outdim = insize - convSize + 1
        hlayer = TanhLayer(outdim * outdim * numFeatureMaps, name='h')
        self.addModule(hlayer)

        outlayer = SigmoidLayer(outdim * outdim, name='out')
        self.addOutputModule(outlayer)

        # build shared weights
        convConns = []
        for i in range(convSize):
            convConns.append(MotherConnection(convSize * numFeatureMaps * inputdim, name='conv' + str(i)))
        outConn = MotherConnection(numFeatureMaps)

        # establish the connections.
        for i in range(outdim):
            for j in range(outdim):
                offset = i * outdim + j
                outmod = ModuleSlice(hlayer, inSliceFrom=offset * numFeatureMaps, inSliceTo=(offset + 1) * numFeatureMaps,
                                     outSliceFrom=offset * numFeatureMaps, outSliceTo=(offset + 1) * numFeatureMaps)
                self.addConnection(SharedFullConnection(outConn, outmod, outlayer, outSliceFrom=offset, outSliceTo=offset + 1))

                for k, mc in enumerate(convConns):
                    offset = insize * (i + k) + j
                    inmod = ModuleSlice(inlayer, outSliceFrom=offset * inputdim, outSliceTo=offset * inputdim + convSize * inputdim)
                    self.addConnection(SharedFullConnection(mc, inmod, outmod))


if __name__ == '__main__':
    from scipy import array, ravel
    from custom.convboard import ConvolutionalBoardNetwork
    from pybrain.rl.environments.twoplayergames.tasks import CaptureGameTask

    N = ConvolutionalBoardNetwork(4, 3, 5)
    input = [[[0, 0], [0, 0], [0, 0], [0, 0]],
             [[0, 0], [0, 0], [0, 0], [1, 1]],
             [[0, 0], [1, 1], [0, 0], [0, 1]],
             [[0, 0], [1, 0], [1, 1], [0, 1]],
             ]
    res = N.activate(ravel(array(input)))
    res = res.reshape(4, 4)
    print(N['pad'].inputbuffer[0].reshape(6, 6, 2)[:, :, 0])
    print(res)

    t = CaptureGameTask(4)
    print(t(N))

    if False:
        N = SimpleConvolutionalNetwork(4, 2, 5)
        print(N)
        res = N.activate(ravel(array(input)))
        res = res.reshape(3, 3)
        print(res)





