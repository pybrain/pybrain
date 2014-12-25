"""

Build a simple recurrent network:
    >>> n = buildRecurrentNetwork()
    >>> print(n)
    RecurrentNetwork
       Modules:
        [<LinearLayer 'in'>, <LinearLayer 'hidden0'>, <LinearLayer 'out'>]
       ...
       Recurrent Connections:
        [<FullConnection ...: 'hidden0' -> 'hidden0'>]

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

Set all the weights to one, and the recurrent one to 0.5, and then do some checks.
    >>> n.params[:] = [1,1,0.5]
    >>> n.reset()
    >>> n.activate(4)[0]
    4.0
    >>> n.activate(-1)[0]
    1.0
    >>> n.activate(0)[0]
    0.5
    >>> n.reset()
    >>> n.activate(0)[0]
    0.0

"""
__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import ones #@UnusedImport
from pybrain import FullConnection
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import LinearLayer
from pybrain.tests import runModuleTestSuite


def buildRecurrentNetwork():
    N = buildNetwork(1, 1, 1, recurrent=True, bias=False, hiddenclass=LinearLayer, outputbias=False)
    h = N['hidden0']
    N.addRecurrentConnection(FullConnection(h, h))
    N.sortModules()
    N.name = 'RecurrentNetwork'
    return N

if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))

