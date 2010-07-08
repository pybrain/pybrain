"""

Build a nested network:

    >>> n = buildNestedNetwork()

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

from pybrain.structure import FeedForwardNetwork
from pybrain import LinearLayer, FullConnection
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tests import runModuleTestSuite


def buildNestedNetwork():
    """ build a nested network. """
    N = FeedForwardNetwork('outer')
    a = LinearLayer(1, name='a')
    b = LinearLayer(2, name='b')
    c = buildNetwork(2, 3, 1)
    c.name = 'inner'
    N.addInputModule(a)
    N.addModule(c)
    N.addOutputModule(b)
    N.addConnection(FullConnection(a, b))
    N.addConnection(FullConnection(b, c))
    N.sortModules()
    return N


if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))

