"""
Trying to build a network with shared connections:

    >>> from random import random
    >>> n = buildSharedCrossedNetwork()

Check if the parameters are the same:

    >>> (n.connections[n['a']][0].params == n.connections[n['a']][1].params).all()
    True

    >>> (n.connections[n['b']][0].params == n.connections[n['c']][0].params).all()
    True

    >>> from pybrain.tools.customxml.networkwriter import NetworkWriter

The transformation of the first input to the second output is identical to the transformation of the
second towards the first:

    >>> r1, r2 = random(), random()
    >>> v1, v2 = n.activate([r1, r2])
    >>> v3, v4 = n.activate([r2, r1])

    >> n['b'].inputbuffer, n['c'].inputbuffer, n['b'].outputbuffer, n['c'].outputbuffer

    >>> v1 == v4
    True
    >>> v2 == v3
    True

Check its gradient:

    >>> from pybrain.tests import gradientCheck
    >>> gradientCheck(n)
    Perfect gradient
    True

Try writing it to an xml file, reread it and determine if it looks the same:

    >>> from pybrain.tests import xmlInvariance
    >>> xmlInvariance(n)
    Same representation
    Same function
    Same class

"""

__author__ = 'Tom Schaul, tom@idsia.ch'

import scipy

from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain import LinearLayer, SharedFullConnection, MotherConnection
from pybrain.tests import runModuleTestSuite


def buildSharedCrossedNetwork():
    """ build a network with shared connections. Two hidden modules are
    symmetrically linked, but to a different input neuron than the
    output neuron. The weights are random. """
    N = FeedForwardNetwork('shared-crossed')
    h = 1
    a = LinearLayer(2, name = 'a')
    b = LinearLayer(h, name = 'b')
    c = LinearLayer(h, name = 'c')
    d = LinearLayer(2, name = 'd')
    N.addInputModule(a)
    N.addModule(b)
    N.addModule(c)
    N.addOutputModule(d)

    m1 = MotherConnection(h)
    m1.params[:] = scipy.array((1,))

    m2 = MotherConnection(h)
    m2.params[:] = scipy.array((2,))

    N.addConnection(SharedFullConnection(m1, a, b, inSliceTo = 1))
    N.addConnection(SharedFullConnection(m1, a, c, inSliceFrom = 1))
    N.addConnection(SharedFullConnection(m2, b, d, outSliceFrom = 1))
    N.addConnection(SharedFullConnection(m2, c, d, outSliceTo = 1))
    N.sortModules()
    return N


if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))

