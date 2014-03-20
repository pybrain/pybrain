__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.structure.modules.module import Module


class NeuronLayer(Module):
    """Module conceptually representing a layer of units """

    # Number of neurons
    dim = 0

    def __init__(self, dim, name=None):
        """Create a layer with dim number of units."""
        Module.__init__(self, dim, dim, name=name)
        self.setArgs(dim=dim)

    def whichNeuron(self, inputIndex=None, outputIndex=None):
        """Determine which neuron a position in the input/output buffer
        corresponds to. """
        if inputIndex is not None:
            return inputIndex
        if outputIndex is not None:
            return outputIndex