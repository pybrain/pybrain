__author__ = 'Michael Isik'


from pybrain.structure.networks.network     import Network
from pybrain.structure.modules.lstm         import LSTMLayer
from pybrain.structure.modules.linearlayer  import LinearLayer
from pybrain.structure.connections.full     import FullConnection
from pybrain.structure.modules.module       import Module
from pybrain.structure.modules.biasunit     import BiasUnit

from numpy import zeros, array, append


class EvolinoNetwork(Module):
    def __init__(self, indim, outdim, hiddim=6):
        Module.__init__(self, indim, outdim)

        self._network = Network()
        self._in_layer = LinearLayer(indim + outdim)
        self._hid_layer = LSTMLayer(hiddim)
        self._out_layer = LinearLayer(outdim)
        self._bias = BiasUnit()

        self._network.addInputModule(self._in_layer)
        self._network.addModule(self._hid_layer)
        self._network.addModule(self._bias)
        self._network.addOutputModule(self._out_layer)


        self._hid_to_out_connection = FullConnection(self._hid_layer , self._out_layer)
        self._in_to_hid_connection = FullConnection(self._in_layer  , self._hid_layer)
        self._network.addConnection(self._hid_to_out_connection)
        self._network.addConnection(self._in_to_hid_connection)
        self._network.addConnection(FullConnection(self._bias, self._hid_layer))

        self._network.sortModules()

        self.offset = self._network.offset
        self.backprojectionFactor = 0.01

    def reset(self):
        self._network.reset()


    def _washout(self, input, target, first_idx=None, last_idx=None):
        assert self.indim == len(input[0])
        assert self.outdim == len(target[0])
        assert len(input) == len(target)

        if first_idx is None: first_idx = 0
        if last_idx  is None: last_idx = len(target) - 1
        raw_outputs = []
        for i in range(first_idx, last_idx + 1):
            backprojection = self._getLastOutput()
            backprojection *= self.backprojectionFactor
            full_inp = self._createFullInput(input[i], backprojection)
            self._activateNetwork(full_inp)
            raw_out = self._getRawOutput()
#            print("RAWOUT: ", full_inp, " --> ", raw_out, self._getLastOutput())
            raw_outputs.append(array(raw_out))
            self._setLastOutput(target[i])

        return array(raw_outputs)



    def _activateNetwork(self, input):
        assert len(input) == self._network.indim
        output = self._network.activate(input)
        self.offset = self._network.offset
#        print("INNNNNNN=", input, "   OUTPP=", output)
        return output

    def activate(self, input):
        assert len(input) == self.indim

        backprojection = self._getLastOutput()
        backprojection *= self.backprojectionFactor
        full_inp = self._createFullInput(input, backprojection)
        out = self._activateNetwork(full_inp)
#        print("AAAAAACT: ", full_inp, "-->", out)

#        self._setLastOutput(last_out*5)

        return out


    def calculateOutput(self, dataset, washout_calculation_ratio=(1, 2)):
        washout_calculation_ratio = array(washout_calculation_ratio, float)
        ratio = washout_calculation_ratio / sum(washout_calculation_ratio)

        # iterate through all sequences
        collected_input = None
        collected_output = None
        collected_target = None
        for i in range(dataset.getNumSequences()):

            seq = dataset.getSequence(i)
            input = seq[0]
            target = seq[1]

            washout_steps = int(len(input) * ratio[0])

            washout_input = input  [               : washout_steps ]
            washout_target = target [               : washout_steps ]
            calculation_target = target [ washout_steps :               ]


            # reset
            self.reset()

            # washout
            self._washout(washout_input, washout_target)


            # collect calculation data
            outputs = []
            inputs = []
#            for i in xrange(washout_steps, len(input)):
            for inp in input[washout_steps:]:
                out = self.activate(inp)
#                    print(out)
#                print(inp)
                inputs.append(inp)
                outputs.append(out)

            # collect output and targets
            if collected_input is not None:
                collected_input = append(collected_input, inputs, axis=0)
            else:
                collected_input = array(inputs)
#            print(collected_input; exit())

            if collected_output is not None:
                collected_output = append(collected_output, outputs, axis=0)
            else:
                collected_output = array(outputs)

            if collected_target is not None:
                collected_target = append(collected_target, calculation_target, axis=0)
            else:
                collected_target = calculation_target

        return collected_input, collected_output, collected_target

    def _createFullInput(self, input, output):
        if self.indim > 0:
            return append(input, output)
        else:
            return array(output)



    def _getLastOutput(self):
        if self.offset == 0:
            return zeros(self.outdim)
        else:
            return self._out_layer.outputbuffer[self.offset - 1]

    def _setLastOutput(self, output):
        self._out_layer.outputbuffer[self.offset - 1][:] = output


    # ======================================================== Genome related ===


    def _validateGenomeLayer(self, layer):
        """ Validates the type and state of a layer
        """
        assert isinstance(layer, LSTMLayer)
        assert not layer.peepholes


    def getGenome(self):
        """ Returns the Genome of the network.
            See class description for more details.
        """
        return self._getGenomeOfLayer(self._hid_layer)


    def setGenome(self, weights):
        """ Sets the Genome of the network.
            See class description for more details.
        """
        weights = deepcopy(weights)
        self._setGenomeOfLayer(self._hid_layer, weights)



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
                    c.params[ cell_idx + 0 * dim ],
                    c.params[ cell_idx + 1 * dim ],
                    c.params[ cell_idx + 2 * dim ],
                    c.params[ cell_idx + 3 * dim ] ]

            layer_weights.append(cell_weights)
        return layer_weights





    def _setGenomeOfLayer(self, layer, weights):
        """ Sets the genome of a single layer.
        """
        self._validateGenomeLayer(layer)

        dim = layer.outdim

        connections = self._getInputConnectionsOfLayer(layer)

        for cell_idx in range(dim):
            cell_weights = weights.pop(0)
            for c in connections:
                params = c.params
                params[cell_idx + 0 * dim] = cell_weights.pop(0)
                params[cell_idx + 1 * dim] = cell_weights.pop(0)
                params[cell_idx + 2 * dim] = cell_weights.pop(0)
                params[cell_idx + 3 * dim] = cell_weights.pop(0)
            assert not len(cell_weights)





    # ============================================ Linear Regression related ===

    def setOutputWeightMatrix(self, W):
        """ Sets the weight matrix of the output layer's input connection.
        """
        c = self._hid_to_out_connection
        c.params[:] = W.flatten()

    def getOutputWeightMatrix(self):
        """ Sets the weight matrix of the output layer's input connection.
        """
        c = self._hid_to_out_connection
        p = c.getParameters()
        return reshape(p, (c.outdim, c.indim))




    def _getRawOutput(self):
        """ Returns the current output of the last hidden layer.
            This is needed for linear regression, which calculates
            the weight matrix W of the full connection between this layer
            and the output layer.
        """
        return copy(self._hid_layer.outputbuffer[self.offset - 1])






    # ====================================================== Topology Helper ===



    def _getInputConnectionsOfLayer(self, layer):
        """ Returns a list of all input connections for the layer. """
        connections = []
        for c in sum(list(self._network.connections.values()), []):
            if c.outmod is layer:
                if not isinstance(c, FullConnection):
                    raise NotImplementedError("At the time there is only support for FullConnection")
                connections.append(c)
        return connections















from numpy import reshape
from copy  import copy, deepcopy


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
        """ :key network: The network to be wrapped
        """
        self.network = network
        self._output_connection = None
        self._last_hidden_layer = None
        self._first_hidden_layer = None
        self._establishRecurrence()

    def getNetwork(self):
        """ Returns the Network """
        return self.network

    def _establishRecurrence(self):
        """ Adds a recurrent full connection from the output layer to the first
            hidden layer.
        """
        network = self.network
        outlayer = self.getOutputLayer()
        hid1layer = self.getFirstHiddenLayer()
        network.addRecurrentConnection(FullConnection(outlayer, hid1layer))


    # ======================================================== Genome related ===


    def _validateGenomeLayer(self, layer):
        """ Validates the type and state of a layer
        """
        assert isinstance(layer, LSTMLayer)
        assert not layer.peepholes


    def getGenome(self):
        """ Returns the Genome of the network.
            See class description for more details.
        """
        weights = []
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

            layer_weights.append(cell_weights)
        return layer_weights





    def _setGenomeOfLayer(self, layer, weights):
        """ Sets the genome of a single layer.
        """
        self._validateGenomeLayer(layer)

        dim = layer.outdim

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
        return reshape(p, (c.outdim, c.indim))


    def injectBackproject(self, injection):
        """ Injects a vector into the recurrent connection.
            This will be used in the evolino trainingsphase, where the target
            values need to be backprojected instead of the real output of the net.

            :key injection: vector of length self.network.outdim
        """
        outlayer = self.getOutputLayer()
        outlayer.outputbuffer[self.network.offset - 1][:] = injection


    def _getRawOutput(self):
        """ Returns the current output of the last hidden layer.
            This is needed for linear regression, which calculates
            the weight matrix W of the full connection between this layer
            and the output layer.
        """
        return copy(self.getLastHiddenLayer().outputbuffer[self.network.offset - 1])


    # ====================================================== Topology Helper ===


    def getOutputLayer(self):
        """ Returns the output layer """
        assert len(self.network.outmodules) == 1
        return self.network.outmodules[0]



    def getOutputConnection(self):
        """ Returns the input connection of the output layer. """
        if self._output_connection is None:
            outlayer = self.getOutputLayer()
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
#                    print(c.inmod)
                    layers.append(c.inmod)

            assert len(layers) == 1
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

            assert len(layers) == 1
            self._first_hidden_layer = layers[0]
        return self._first_hidden_layer



    def getConnections(self):
        """ Returns a list of all connections. """
        return sum(list(self.network.connections.values()), [])

    def getInputLayer(self):
        """ Returns the input layer. """
        assert len(self.network.inmodules) == 1
        return self.network.inmodules[0]

    def _getInputConnectionsOfLayer(self, layer):
        """ Returns a list of all input connections for the layer. """
        connections = []
        for c in sum(list(self.network.connections.values()), []):
            if c.outmod is layer:
                if not isinstance(c, FullConnection):
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





