"""

Trying to build a network with shared connections:

    >>> from random import random
    >>> n = buildSharedCrossedNetwork()
    
    >>> from pybrain.tools.xml.networkwriter import NetworkWriter
    
The transfomation of the first input to the second output is identical to the transformation of the 
second towards the first:

    >>> r1, r2 = random(), random()
    >>> v1, v2 = n.activate([r1, r2])
    >>> v3, v4 = n.activate([r2, r1])
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

from pybrain import Network, LinearLayer, SigmoidLayer, SharedFullConnection, MotherConnection
from pybrain.tests import runModuleTestSuite


def buildSharedCrossedNetwork():
    """ build a network with shared connections. Two hiddne modules are symetrically linked, but to a different 
    input neuron than the output neuron. The weights are random. """
    N = Network('shared-crossed')
    h = 3
    a = LinearLayer(2, name = 'a')
    b = SigmoidLayer(h, name = 'b')
    c = SigmoidLayer(h, name = 'c')
    d = LinearLayer(2, name = 'd')
    N.addInputModule(a)
    N.addModule(b)
    N.addModule(c)
    N.addOutputModule(d)
    
    m1 = MotherConnection(h)
    m2 = MotherConnection(h)
    
    N.addConnection(SharedFullConnection(m1, a, b, inSliceTo = 1))
    N.addConnection(SharedFullConnection(m1, a, c, inSliceFrom = 1))
    N.addConnection(SharedFullConnection(m2, b, d, outSliceFrom = 1))
    N.addConnection(SharedFullConnection(m2, c, d, outSliceTo = 1))
    N.sortModules()
    return N


if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))

