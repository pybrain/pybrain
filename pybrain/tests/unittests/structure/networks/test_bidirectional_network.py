"""

Build a bi-directional Network for sequences (each sample a single value) of length 20:

    >>> n = BidirectionalNetwork(seqlen=20, inputsize=1, hiddensize=5, symmetric=False)

It should have 2x1x5 + 2x1x5 + 2x5x5 = 70 weights

    >>> n.paramdim
    70

Now let's build a symmetric network:

    >>> n = BidirectionalNetwork(seqlen=12, inputsize=2, hiddensize=3, symmetric=True)
    >>> n.indim
    24

It should have 1x2x3 + 1x1x3 + 1x3x3 = 18 weights

    >>> n.paramdim
    18

A forward pass:

    >>> from numpy import ones
    >>> r = n.activate(ones(24))
    >>> len(r)
    12

The result should be symmetric (although the weights are random)

    >>> r[0]-r[-1]
    0.0

Check its gradient:

    >>> from pybrain.tests import gradientCheck
    >>> gradientCheck(n)
    Perfect gradient
    True


"""

__author__ = 'Tom Schaul, tom@idsia.ch'


from pybrain.structure.networks import BidirectionalNetwork #@UnusedImport
from pybrain.tests import runModuleTestSuite


if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))

