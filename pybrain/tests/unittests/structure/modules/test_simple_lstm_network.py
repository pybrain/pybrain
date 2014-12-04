"""

Build a simple lstm network with peepholes:
    >>> n = buildSimpleLSTMNetwork(True)
    >>> print(n)
    simpleLstmNet
       Modules:
        [<BiasUnit 'bias'>, <LinearLayer 'i'>, <LSTMLayer 'lstm'>, <LinearLayer 'o'>]
       Connections:
        [<FullConnection 'f1': 'i' -> 'lstm'>, <FullConnection 'f2': 'bias' -> 'lstm'>, <FullConnection 'r1': 'lstm' -> 'o'>]
       Recurrent Connections:
        [<FullConnection 'r1': 'lstm' -> 'lstm'>]

Check its gradient:

    >>> from pybrain.tests import gradientCheck
    >>> gradientCheck(n)
    Perfect gradient
    True

    >>> net = RecurrentNetwork()
    >>> l = LSTMLayer(1)
    >>> net.addRecurrentConnection(FullConnection(l, l))
    >>> net.addInputModule(l)
    >>> net.outmodules.append(l)
    >>> net.sortModules()
    >>> gradientCheck(net)
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

from pybrain.structure.networks.recurrent import RecurrentNetwork
from pybrain import LinearLayer, FullConnection, LSTMLayer, BiasUnit
from pybrain.tests import runModuleTestSuite


def buildSimpleLSTMNetwork(peepholes = False):
    N = RecurrentNetwork('simpleLstmNet')
    i = LinearLayer(1, name = 'i')
    h = LSTMLayer(1, peepholes = peepholes, name = 'lstm')
    o = LinearLayer(1, name = 'o')
    b = BiasUnit('bias')
    N.addModule(b)
    N.addOutputModule(o)
    N.addInputModule(i)
    N.addModule(h)
    N.addConnection(FullConnection(i, h, name = 'f1'))
    N.addConnection(FullConnection(b, h, name = 'f2'))
    N.addRecurrentConnection(FullConnection(h, h, name = 'r1'))
    N.addConnection(FullConnection(h, o, name = 'r1'))
    N.sortModules()
    return N


if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))

