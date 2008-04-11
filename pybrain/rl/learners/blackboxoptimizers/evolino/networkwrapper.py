
__author__ = 'Michael Isik'



from pybrain.structure.networks.network     import Network
from pybrain.structure.modules.lstm         import LSTMLayer
from pybrain.structure.modules.linearlayer  import LinearLayer
from pybrain.structure.connections.full     import FullConnection
from pybrain.structure.connections.identity import IdentityConnection

from numpy import reshape
from copy  import copy,deepcopy



class NetworkWrapper(object):
    """ Network wrapper class for Evolino Networks

        This class implements methods for extracting and setting the genome of
        the supplied network to allow its evolving.
        The genome of the network consists of the input weights of each hidden
        lstm neuron. The structure of the genome will be a list of lists,
        where the inner lists bundle all input weights of on neuron:
            [ [ neuron1's inweights ] , [ neuron2's inweights ] , ... ]
        The inner lists will be used as chromosomes inside the evolino framework.

        Also there are methods that help with the linear regression part.
        They can extract end set the weight matrix W for the last full-connection.

        At the moment the network must meet following constraints:
            - All hidden layers that have input connections must be of type LSTMLayer
            - The LSTMLayer do not use peepholes
            - There must be exactly one output-layer
            - There must be exactly one input-layer
            - There must be only one layer, that is connected to the output layer
            - The input layer must be connected to only one hidden layer
            - All used connections must be of type FullConnection

        When the network is supplied it will be augmented with a
        recurrent full connection from the output layer to the first hidden layer.
        So do not do this yourself.

    """
    def __init__(self, network):
        """ @param network: The network to be wrapped
        """
        self.network = network
        self._output_connection  = None
        self._last_hidden_layer  = None
        self._first_hidden_layer = None
        self._establishRecurrence()

    def getNetwork(self):
        """ Returns the Network """
        return self.network

    def _establishRecurrence(self):
        """ Adds a recurrent full connection from the output layer to the first
            hidden layer.
        """
        network   = self.network
        outlayer  = self.getOutputLayer()
        hid1layer = self.getFirstHiddenLayer()
        network.addRecurrentConnection( FullConnection( outlayer, hid1layer ) )


    # ======================================================== Genome related ===


    def _validateGenomeLayer(self, layer):
        """ Validates the type and state of a layer
        """
        assert isinstance(layer,LSTMLayer)
        assert not layer.peepholes


    def getGenome(self):
        """ Returns the Genome of the network.
            See class description for more details.
        """
        weights=[]
        network = self.network
        for layer in self.getHiddenLayers():
            if isinstance(layer, LSTMLayer):
#                 if layer is not self._recurrence_layer:
                weights += self._getGenomeOfLayer(layer)
        return weights

    def setGenome(self, weights):
        """ Sets the Genome of the network.
            See class description for more details.
        """
        weights = deepcopy(weights)
        for layer in self.getHiddenLayers():
            if isinstance(layer, LSTMLayer):
#               if layer is not self._recurrence_layer:
                self._setGenomeOfLayer(layer, weights)



    def _getGenomeOfLayer(self, layer):
        """ Returns the genome of a single layer.
        """
        self._validateGenomeLayer(layer)

        dim = layer.outdim
        layer_weights = []

        connections = self._getInputConnectionsOfLayer(layer)

        for cell_idx in range(dim):
            # todo: the evolino paper uses a different order of weights for the genotype of a lstm cell
            cell_weights = []
            for c in connections:
                cell_weights += [
                    c.getParameters()[ cell_idx + 0 * dim ],
                    c.getParameters()[ cell_idx + 1 * dim ],
                    c.getParameters()[ cell_idx + 2 * dim ],
                    c.getParameters()[ cell_idx + 3 * dim ] ]

            layer_weights.append( cell_weights )
        return layer_weights





    def _setGenomeOfLayer(self, layer, weights):
        """ Sets the genome of a single layer.
        """
        self._validateGenomeLayer(layer)

        dim = layer.outdim
        layer_weights = []

        connections = self._getInputConnectionsOfLayer(layer)

        for cell_idx in range(dim):
            cell_weights = weights.pop(0)
            for c in connections:
                params = c.getParameters()
                params[cell_idx + 0 * dim] = cell_weights.pop(0)
                params[cell_idx + 1 * dim] = cell_weights.pop(0)
                params[cell_idx + 2 * dim] = cell_weights.pop(0)
                params[cell_idx + 3 * dim] = cell_weights.pop(0)
            assert not len(cell_weights)



    # ============================================ Linear Regression related ===

    def setOutputWeightMatrix(self, W):
        """ Sets the weight matrix of the output layer's input connection.
        """
        c = self.getOutputConnection()
        p = c.getParameters()
        p[:] = W.flatten()

    def getOutputWeightMatrix(self):
        """ Sets the weight matrix of the output layer's input connection.
        """
        c = self.getOutputConnection()
        p = c.getParameters()
        return reshape( p, (c.outdim, c.indim) )


    def injectBackproject(self, injection):
        """ Injects a vector into the recurrent connection.
            This will be used in the evolino trainingsphase, where the target
            values need to be backprojected instead of the real output of the net.
            @param injection: vector of length self.network.outdim
        """
        outlayer = self.getOutputLayer()
        outlayer.outputbuffer[self.network.time-1][:] = injection


    def _getRawOutput(self):
        """ Returns the current output of the last hidden layer.
            This is needed for linear regression, which calculates
            the weight matrix W of the full connection between this layer
            and the output layer.
        """
        return copy(self.getLastHiddenLayer().outputbuffer[self.network.time-1])


    # ====================================================== Topology Helper ===


    def getOutputLayer(self):
        """ Returns the output layer """
        assert len(self.network.outmodules)==1
        return self.network.outmodules[0]



    def getOutputConnection(self):
        """ Returns the input connection of the output layer. """
        if self._output_connection is None:
            outlayer  = self.getOutputLayer()
            lastlayer = self.getLastHiddenLayer()
            for c in self.getConnections():
                if c.outmod is outlayer:
                    assert c.inmod is lastlayer
                    self._output_connection = c

        return self._output_connection



    def getLastHiddenLayer(self):
        """ Returns the last hidden layer. """
        if self._last_hidden_layer is None:
            outlayer = self.getOutputLayer()
            layers = []
            for c in self.getConnections():
                if c.outmod is outlayer:
                    print c.inmod
                    layers.append(c.inmod)

            assert len(layers)==1
            self._last_hidden_layer = layers[0]
        return self._last_hidden_layer



    def getFirstHiddenLayer(self):
        """ Returns the first hidden layer. """
        if self._first_hidden_layer is None:
            inlayer = self.getInputLayer()
            layers = []
            for c in self.getConnections():
                if c.inmod is inlayer:
                    layers.append(c.outmod)

            assert len(layers)==1
            self._first_hidden_layer = layers[0]
        return self._first_hidden_layer



    def getConnections(self):
        """ Returns a list of all connections. """
        return sum( self.network.connections.values(), [] )

    def getInputLayer(self):
        """ Returns the input layer. """
        assert len(self.network.inmodules)==1
        return self.network.inmodules[0]

    def _getInputConnectionsOfLayer(self, layer):
        """ Returns a list of all input connections for the layer. """
        connections = []
        for c in sum( self.network.connections.values(), [] ):
            if c.outmod is layer:
                if not isinstance( c, FullConnection ):
                    raise NotImplementedError("At the time there is only support for FullConnection")
                connections.append(c)
        return connections



    def getHiddenLayers(self):
        """ Returns a list of all hidden layers. """
        layers = []
        network = self.network
        for m in network.modules:
             if m not in network.inmodules and m not in network.outmodules:
                 layers.append(m)
        return layers





