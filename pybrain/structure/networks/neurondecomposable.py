__author__ = 'Daan Wierstra and Tom Schaul'

from itertools import chain

from scipy import zeros

from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.structure.networks.recurrent import RecurrentNetwork
from pybrain.structure.modules.neuronlayer import NeuronLayer
from pybrain.structure.connections import FullConnection

# CHECKME: allow modules that do not inherit from NeuronLayer? and treat them as single neurons?


class NeuronDecomposableNetwork(object):
    """ A Network, that allows accessing parameters decomposed by their
    corresponding individual neuron. """

    # ESP style treatment:
    espStyleDecomposition = True

    def addModule(self, m):
        assert isinstance(m, NeuronLayer)
        super(NeuronDecomposableNetwork, self).addModule(m)

    def sortModules(self):
        super(NeuronDecomposableNetwork, self).sortModules()
        self._constructParameterInfo()

        # contains a list of lists of indices
        self.decompositionIndices = {}
        for neuron in self._neuronIterator():
            self.decompositionIndices[neuron] = []
        for w in range(self.paramdim):
            inneuron, outneuron = self.paramInfo[w]
            if self.espStyleDecomposition and outneuron[0] in self.outmodules:
                self.decompositionIndices[inneuron].append(w)
            else:
                self.decompositionIndices[outneuron].append(w)

    def _neuronIterator(self):
        for m in self.modules:
            for n in range(m.dim):
                yield (m, n)

    def _constructParameterInfo(self):
        """ construct a dictionnary with information about each parameter:
        The key is the index in self.params, and the value is a tuple containing
        (inneuron, outneuron), where a neuron is a tuple of it's module and an index.
        """
        self.paramInfo = {}
        index = 0
        for x in self._containerIterator():
            if isinstance(x, FullConnection):
                for w in range(x.paramdim):
                    inbuf, outbuf = x.whichBuffers(w)
                    self.paramInfo[index + w] = ((x.inmod, x.inmod.whichNeuron(outputIndex=inbuf)),
                                               (x.outmod, x.outmod.whichNeuron(inputIndex=outbuf)))
            elif isinstance(x, NeuronLayer):
                for n in range(x.paramdim):
                    self.paramInfo[index + n] = ((x, n), (x, n))
            else:
                raise
            index += x.paramdim

    def getDecomposition(self):
        """ return a list of arrays, each corresponding to one neuron's relevant parameters """
        res = []
        for neuron in self._neuronIterator():
            nIndices = self.decompositionIndices[neuron]
            if len(nIndices) > 0:
                tmp = zeros(len(nIndices))
                for i, ni in enumerate(nIndices):
                    tmp[i] = self.params[ni]
                res.append(tmp)
        return res

    def setDecomposition(self, decomposedParams):
        """ set parameters by neuron decomposition,
        each corresponding to one neuron's relevant parameters """
        nindex = 0
        for neuron in self._neuronIterator():
            nIndices = self.decompositionIndices[neuron]
            if len(nIndices) > 0:
                for i, ni in enumerate(nIndices):
                    self.params[ni] = decomposedParams[nindex][i]
                nindex += 1

    @staticmethod
    def convertNormalNetwork(n):
        """ convert a normal network into a decomposable one """
        if isinstance(n, RecurrentNetwork):
            res = RecurrentDecomposableNetwork()
            for c in n.recurrentConns:
                res.addRecurrentConnection(c)
        else:
            res = FeedForwardDecomposableNetwork()
        for m in n.inmodules:
            res.addInputModule(m)
        for m in n.outmodules:
            res.addOutputModule(m)
        for m in n.modules:
            res.addModule(m)
        for c in chain(*n.connections.values()):
            res.addConnection(c)
        res.name = n.name
        res.sortModules()
        return res


class FeedForwardDecomposableNetwork(NeuronDecomposableNetwork, FeedForwardNetwork):
    pass


class RecurrentDecomposableNetwork(NeuronDecomposableNetwork, RecurrentNetwork):
    pass
