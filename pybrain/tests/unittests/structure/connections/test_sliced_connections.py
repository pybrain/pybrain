"""
    >>> from scipy import array
    >>> from pybrain.tests import epsilonCheck

Trying to build a network with shared connections:

    >>> from random import random
    >>> n = buildSlicedNetwork()
    >>> n.params[:] = array((2, 2))

The transformation of the first input to the second output is identical to the transformation of the
second towards the first:

    >>> r1, r2 = 2.5, 3.2
    >>> v1, v2 = n.activate([r1, r2])
    >>> epsilonCheck(6.4 - v1)
    True
    >>> epsilonCheck(5 - v2)
    True


"""

__author__ = 'Tom Schaul, tom@idsia.ch'


from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain import LinearLayer, FullConnection
from pybrain.tests import runModuleTestSuite


def buildSlicedNetwork():
    """ build a network with shared connections. Two hidden modules are
    symmetrically linked, but to a different input neuron than the
    output neuron. The weights are random. """
    N = FeedForwardNetwork('sliced')
    a = LinearLayer(2, name = 'a')
    b = LinearLayer(2, name = 'b')
    N.addInputModule(a)
    N.addOutputModule(b)

    N.addConnection(FullConnection(a, b, inSliceTo=1, outSliceFrom=1))
    N.addConnection(FullConnection(a, b, inSliceFrom=1, outSliceTo=1))
    N.sortModules()
    return N


if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))
