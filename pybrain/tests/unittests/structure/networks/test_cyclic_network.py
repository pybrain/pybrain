"""

Trying to build a cyclic network (should fail):

    >>> buildCyclicNetwork(False)
    Traceback (most recent call last):
        ...
    NetworkConstructionException: Loop in network graph.

If one connection is recurrent, it should work:

    >>> buildCyclicNetwork(True)
    <RecurrentNetwork 'cyc'>

"""

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain import FeedForwardNetwork, RecurrentNetwork, LinearLayer, \
    FullConnection
from pybrain.tests import runModuleTestSuite


def buildCyclicNetwork(recurrent):
    """ build a cyclic network with 4 modules

    :key recurrent: make one of the connections recurrent """
    Network = RecurrentNetwork if recurrent else FeedForwardNetwork
    N = Network('cyc')
    a = LinearLayer(1, name='a')
    b = LinearLayer(2, name='b')
    c = LinearLayer(3, name='c')
    d = LinearLayer(4, name='d')
    N.addInputModule(a)
    N.addModule(b)
    N.addModule(d)
    N.addOutputModule(c)
    N.addConnection(FullConnection(a, b))
    N.addConnection(FullConnection(b, c))
    N.addConnection(FullConnection(c, d))
    if recurrent:
        N.addRecurrentConnection(FullConnection(d, a))
    else:
        N.addConnection(FullConnection(d, a))
    N.sortModules()
    return N


if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))

