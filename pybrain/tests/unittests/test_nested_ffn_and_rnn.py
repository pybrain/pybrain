"""

Build a mixed nested network:

    >>> n = buildMixedNestedNetwork()
    >>> inner = n['inner']

Some specific tests:
The feed-forward part should have its buffers increased in size, and
keep the correct offset.

    >>> len(inner.outputbuffer)
    1

    >>> o1 = n.activate([1])
    >>> o2 = n.activate([2])
    >>> o2 = n.activate([3])
    >>> (o1 == o2).any()
    False

    >>> n.offset
    3
    >>> inner.offset
    3
    >>> len(inner.outputbuffer)
    4

Verify everything is still fine after reset
    >>> n.reset()

    >>> n.offset
    0
    >>> inner.offset
    0



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

from pybrain.structure import RecurrentNetwork
from pybrain import LinearLayer, FullConnection
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tests import runModuleTestSuite


def buildMixedNestedNetwork():
    """ build a nested network with the inner one being a ffn and the outer one being recurrent. """
    N = RecurrentNetwork('outer')
    a = LinearLayer(1, name='a')
    b = LinearLayer(2, name='b')
    c = buildNetwork(2, 3, 1)
    c.name = 'inner'
    N.addInputModule(a)
    N.addModule(c)
    N.addOutputModule(b)
    N.addConnection(FullConnection(a, b))
    N.addConnection(FullConnection(b, c))
    N.addRecurrentConnection(FullConnection(c, c))
    N.sortModules()
    return N


if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))

