"""
    >>> from pybrain.tests import epsilonCheck
    >>> n = buildSubsamplingNetwork()

All those inputs will be averaged in two blocks (first 4 and last 2),
so they should produce the same outputs.

    >>> x1 = n.activate([3,0,0,0,0,2])[0]
    >>> x2 = n.activate([0,0,0,3,2,0])[0]
    >>> x3 = n.activate([1,1,-2,3,1,1])[0]

    >>> epsilonCheck(x1 - x2)
    True
    >>> epsilonCheck(x1 - x3)
    True


"""

__author__ = 'Tom Schaul, tom@idsia.ch'


from pybrain.structure.connections.subsampling import SubsamplingConnection
from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain import LinearLayer
from pybrain.tests import runModuleTestSuite


def buildSubsamplingNetwork():
    """ Builds a network with subsampling connections. """
    n = FeedForwardNetwork()
    n.addInputModule(LinearLayer(6, 'in'))
    n.addOutputModule(LinearLayer(1, 'out'))
    n.addConnection(SubsamplingConnection(n['in'], n['out'], inSliceTo=4))
    n.addConnection(SubsamplingConnection(n['in'], n['out'], inSliceFrom=4))
    n.sortModules()
    return n


if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))
