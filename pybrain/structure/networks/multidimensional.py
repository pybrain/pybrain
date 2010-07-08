__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.structure.networks.swiping import SwipingNetwork
from pybrain import MDLSTMLayer, IdentityConnection
from pybrain import ModuleMesh, LinearLayer, TanhLayer, SigmoidLayer
from scipy import product


class MultiDimensionalRNN(SwipingNetwork):
    """ One possible implementation of Multi-dimensional Recurrent Neural Networks."""

    insize = 1
    outputs = 1
    hsize = 5
    componentclass = TanhLayer
    outcomponentclass = SigmoidLayer

    def __init__(self, dims, **args):
        """ The one required argument specifies the sizes of each dimension (minimum 2) """

        SwipingNetwork.__init__(self, dims = dims, **args)

        pdims = product(dims)
        # the input is a 2D-mesh (as a view on a flat input layer)
        inmod = LinearLayer(self.insize*pdims, name = 'input')
        inmesh = ModuleMesh.viewOnFlatLayer(inmod, dims, 'inmesh')

        # the output is a 2D-mesh (as a view on a flat sigmoid output layer)
        outmod = self.outcomponentclass(self.outputs*pdims, name = 'output')
        outmesh = ModuleMesh.viewOnFlatLayer(outmod, dims, 'outmesh')

        if self.componentclass is MDLSTMLayer:
            c = lambda: MDLSTMLayer(self.hsize, 2, self.peepholes).meatSlice()
            adims = tuple(list(dims)+[4])
            hiddenmesh = ModuleMesh(c, adims, 'hidden', baserename = True)
        else:
            hiddenmesh = ModuleMesh.constructWithLayers(self.componentclass, self.hsize, tuple(list(dims)+[self.swipes]), 'hidden')

        self._buildSwipingStructure(inmesh, hiddenmesh, outmesh)

        # add the identity connections for the states
        for m in self.modules:
            if isinstance(m, MDLSTMLayer):
                tmp = m.stateSlice()
                index = 0
                for c in list(self.connections[m]):
                    if isinstance(c.outmod, MDLSTMLayer):
                        self.addConnection(IdentityConnection(tmp, c.outmod.stateSlice(),
                                                              outSliceFrom = self.hsize*(index),
                                                              outSliceTo = self.hsize*(index+1)))
                        index += 1

        self.sortModules()

class MultiDimensionalLSTM(MultiDimensionalRNN):
    """ The same, but with LSTM cells in the hidden layer. """
    componentclass = MDLSTMLayer
    peepholes = False

