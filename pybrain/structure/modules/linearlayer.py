__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.structure.modules.neuronlayer import NeuronLayer


class LinearLayer(NeuronLayer):
    """ The simplest kind of module, not doing any transformation. 

    Does not inherit ParameterContainer interface.
    """

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = inbuf

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:] = outerr