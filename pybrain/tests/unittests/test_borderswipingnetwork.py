""" A few tests for BorderSwipingNetworks
    >>> from pybrain import MotherConnection
    >>> from scipy import ones, array

We will use a simple 3-dimensional network:

    >>> dim = 3
    >>> size = 3
    >>> hsize = 1

It is possible to define some weights before construction:

    >>> predefined = {'outconn': MotherConnection(1)}
    >>> predefined['outconn']._setParameters([0.5])

Building it with the helper function below:

    >>> net = buildSimpleBorderSwipingNet(size, dim, hsize, predefined)
    >>> net.name
    'BorderSwipingNetwork-...

    >>> net.paramdim
    7

    >>> net.dims
    (3, 3, 3)


Did the weight get set correctly?

    >>> net.params[0]
    0.5

Now we'll set all weights to a sequence of values:

    >>> net._setParameters(array(range(net.paramdim))/10.+.1)
    >>> nearlyEqual(list(net.params), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    True

Now we want to use the same weights to build a bigger network

    >>> size2 = size + 2
    >>> net2 = buildSimpleBorderSwipingNet(size2, dim, hsize, net.predefined)

It has a few more parameters:

    >>> net2.paramdim
    12

But the values are the same than before except numerical differences.

    >>> nearlyEqual(list(net2.params), [0.1, 0.2, 0.3, 0.3, 0.4, 0.5, 0.4333, 0.40, 0.46666, 0.4142857, 0.6, 0.7])
    True

Let's attempt a couple of activations:

    >>> res = net.activate(array(range(net.indim))/10.)
    >>> res2 = net2.activate(array(range(net2.indim))/10.)
    >>> min(res), min(res2)
    (0.625..., 0.631...)

    >>> max(res), max(res2)
    (0.797..., 0.7999...)


"""

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.tests import runModuleTestSuite
from pybrain.structure.networks import BorderSwipingNetwork
from pybrain import ModuleMesh, LinearLayer, TanhLayer


def nearlyEqual(lst1, lst2, tolerance=0.001):
    """Tell whether the itemwise differences of the two lists is never bigger
    than tolerance."""
    return all(abs(i - j) <= tolerance for i, j in zip(lst1, lst2))


def buildSimpleBorderSwipingNet(size = 3, dim = 3, hsize = 1, predefined = {}):
    """ build a simple swiping network,of given size and dimension, using linear inputs and output"""
    # assuming identical size in all dimensions
    dims = tuple([size]*dim)
    # also includes one dimension for the swipes
    hdims = tuple(list(dims)+[2**dim])
    inmod = LinearLayer(size**dim, name = 'input')
    inmesh = ModuleMesh.viewOnFlatLayer(inmod, dims, 'inmesh')
    outmod = LinearLayer(size**dim, name = 'output')
    outmesh = ModuleMesh.viewOnFlatLayer(outmod, dims, 'outmesh')
    hiddenmesh = ModuleMesh.constructWithLayers(TanhLayer, hsize, hdims, 'hidden')
    return BorderSwipingNetwork(inmesh, hiddenmesh, outmesh, predefined = predefined)


if __name__ == '__main__':
    runModuleTestSuite(__import__('__main__'))
