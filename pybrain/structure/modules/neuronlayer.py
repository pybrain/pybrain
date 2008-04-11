__author__ = 'Tom Schaul, tom@idsia.ch'

from module import Module


class NeuronLayer(Module):
    """ A module conceptually representing a layer of units """
    
    # number of neurons
    dim = 0
    
    def __init__(self, dim, name = None):
        """ @param dim: number of units """
        Module.__init__(self, dim, dim, name = name)
        self.setArgs(dim = dim)
        
    def whichNeuron(self, inputIndex = None, outputIndex = None):
        """ determine which neuron a position in the input/output buffer corresponds to. """
        if inputIndex != None:
            return inputIndex
        if outputIndex != None:
            return outputIndex