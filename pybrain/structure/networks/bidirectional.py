__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.structure.modules import TanhLayer, SigmoidLayer
from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.structure.connections.shared import MotherConnection, SharedFullConnection
from pybrain.structure.modules.linearlayer import LinearLayer
from pybrain.structure.modulemesh import ModuleMesh


class BidirectionalNetwork(FeedForwardNetwork):
    """ A bi-directional recurrent neural network, implemented as unfolded in time. """

    #: should the weights for the forward-direction be the same than for the backward-direction?
    symmetric = False

    #: class for the hidden layers
    componentclass = TanhLayer

    #: class for the output layers
    outcomponentclass = SigmoidLayer

    #: number of inputs for each component of the sequence
    inputsize = 1

    #: number of outputs for each component of the sequence
    outputsize = 1

    #: number of hidden neurons in each hiddne layer
    hiddensize = 5

    #: length of the sequences
    seqlen = None

    def __init__(self, predefined = None, **kwargs):
        """ For the current implementation, the sequence length
        needs to be fixed, and given at construction time. """
        if predefined is not None:
            self.predefined = predefined
        else:
            self.predefined = {}
        FeedForwardNetwork.__init__(self, **kwargs)
        assert self.seqlen is not None

        # the input is a 1D-mesh (as a view on a flat input layer)
        inmod = LinearLayer(self.inputsize * self.seqlen, name='input')
        inmesh = ModuleMesh.viewOnFlatLayer(inmod, (self.seqlen,), 'inmesh')

        # the output is also a 1D-mesh
        outmod = self.outcomponentclass(self.outputsize * self.seqlen, name='output')
        outmesh = ModuleMesh.viewOnFlatLayer(outmod, (self.seqlen,), 'outmesh')

        # the hidden layers are places in a 2xseqlen mesh
        hiddenmesh = ModuleMesh.constructWithLayers(self.componentclass, self.hiddensize,
                                                    (2, self.seqlen), 'hidden')

        # add the modules
        for c in inmesh:
            self.addInputModule(c)
        for c in outmesh:
            self.addOutputModule(c)
        for c in hiddenmesh:
            self.addModule(c)

        # set the connections weights to be shared
        inconnf = MotherConnection(inmesh.componentOutdim * hiddenmesh.componentIndim, name='inconn')
        outconnf = MotherConnection(outmesh.componentIndim * hiddenmesh.componentOutdim, name='outconn')
        forwardconn = MotherConnection(hiddenmesh.componentIndim * hiddenmesh.componentOutdim, name='fconn')
        if self.symmetric:
            backwardconn = forwardconn
            inconnb = inconnf
            outconnb = outconnf
        else:
            backwardconn = MotherConnection(hiddenmesh.componentIndim * hiddenmesh.componentOutdim, name='bconn')
            inconnb = MotherConnection(inmesh.componentOutdim * hiddenmesh.componentIndim, name='inconn')
            outconnb = MotherConnection(outmesh.componentIndim * hiddenmesh.componentOutdim, name='outconn')

        # build the connections
        for i in range(self.seqlen):
            # input to hidden
            self.addConnection(SharedFullConnection(inconnf, inmesh[(i,)], hiddenmesh[(0, i)]))
            self.addConnection(SharedFullConnection(inconnb, inmesh[(i,)], hiddenmesh[(1, i)]))
            # hidden to output
            self.addConnection(SharedFullConnection(outconnf, hiddenmesh[(0, i)], outmesh[(i,)]))
            self.addConnection(SharedFullConnection(outconnb, hiddenmesh[(1, i)], outmesh[(i,)]))
            if i > 0:
                # forward in time
                self.addConnection(SharedFullConnection(forwardconn, hiddenmesh[(0, i - 1)], hiddenmesh[(0, i)]))
            if i < self.seqlen - 1:
                # backward in time
                self.addConnection(SharedFullConnection(backwardconn, hiddenmesh[(1, i + 1)], hiddenmesh[(1, i)]))

        self.sortModules()


