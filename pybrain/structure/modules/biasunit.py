__author__ = 'Tom Schaul, tom@idsia.ch'

from neuronlayer import NeuronLayer
from module import Module


class BiasUnit(NeuronLayer):
    """A simple bias unit with a single constant output."""
    
    dim = 1
    
    def __init__(self, name=None):
        Module.__init__(self, 0, 1, name = name)
        
    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = 1    