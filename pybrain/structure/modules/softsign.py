__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.structure.modules.neuronlayer import NeuronLayer

class SoftSignLayer(NeuronLayer):
    """ softsign activation function as described in X. Glorot and Y.
        Bengio. Understanding the difficulty of training deep feedforward neural
        networks. In Proceedings of the 13th International Workshop on
        Artificial Intelligence and Statistics, 2010. """

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = inbuf / (1 + abs(inbuf))

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:] = (1 - abs(outbuf))**2 * outerr
