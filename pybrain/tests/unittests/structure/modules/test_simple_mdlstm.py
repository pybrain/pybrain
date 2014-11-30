"""

Build a simple mdlstm network with peepholes:
    >>> n = buildSimpleMDLSTMNetwork(True)
    >>> print(n)
    simpleMDLstmNet
       Modules:
        [<BiasUnit 'bias'>, <LinearLayer 'i'>, <MDLSTMLayer 'MDlstm'>, <LinearLayer 'o'>]
       Connections:
        [<FullConnection 'f1': 'i' -> 'MDlstm'>, <FullConnection 'f2': 'bias' -> 'MDlstm'>, <FullConnection 'f3': 'MDlstm' -> 'o'>]
       Recurrent Connections:
        [<FullConnection 'r1': 'MDlstm' -> 'MDlstm'>, <IdentityConnection 'rstate': 'MDlstm' -> 'MDlstm'>]

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

from pybrain.structure.networks.recurrent import RecurrentNetwork
from pybrain import LinearLayer, FullConnection, MDLSTMLayer, BiasUnit, IdentityConnection
from pybrain.tests import runModuleTestSuite


def buildSimpleMDLSTMNetwork(peepholes = False):
    N = RecurrentNetwork('simpleMDLstmNet')
    i = LinearLayer(1, name = 'i')
    dim = 1
    h = MDLSTMLayer(dim, peepholes = peepholes, name = 'MDlstm')
    o = LinearLayer(1, name = 'o')
    b = BiasUnit('bias')
    N.addModule(b)
    N.addOutputModule(o)
    N.addInputModule(i)
    N.addModule(h)
    N.addConnection(FullConnection(i, h, outSliceTo = 4*dim, name = 'f1'))
    N.addConnection(FullConnection(b, h, outSliceTo = 4*dim, name = 'f2'))
    N.addRecurrentConnection(FullConnection(h, h, inSliceTo = dim, outSliceTo = 4*dim, name = 'r1'))
    N.addRecurrentConnection(IdentityConnection(h, h, inSliceFrom = dim, outSliceFrom = 4*dim, name = 'rstate'))
    N.addConnection(FullConnection(h, o, inSliceTo = dim, name = 'f3'))
    N.sortModules()
    return N


if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))

