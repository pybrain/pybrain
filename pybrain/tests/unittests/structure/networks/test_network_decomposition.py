"""

Build a decomposable network
    >>> n = buildDecomposableNetwork()

Check if it was built correctly
    >>> print(n.paramdim)
    12
    >>> tmp = n.getDecomposition()
    >>> tmp[2]
    array([ 1.,  1.,  1.,  1.])

Let's keep the output value for later
    >>> act = n.activate([-1.2,0.5])


Now, change the values for the first neuron
    >>> tmp[0] *= 0

The network has not changed yet
    >>> n.getDecomposition()[0]
    array([ 1.,  1.,  1.,  1.])

Now it has:
    >>> n.setDecomposition(tmp)
    >>> n.getDecomposition()[0]
    array([ 0.,  0.,  0.,  0.])

The new output value should be 2/3 of the original one, with one neuron disabled.

    >>> act2 = n.activate([-1.2,0.5])
    >>> (act2 * 3 / 2 - act)[0]
    0.0

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

from scipy import ones

from pybrain.structure.networks import NeuronDecomposableNetwork
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tests import runModuleTestSuite


def buildDecomposableNetwork():
    """ three hidden neurons, with 2 in- and 2 outconnections each. """
    n = buildNetwork(2, 3, 2, bias = False)
    ndc = NeuronDecomposableNetwork.convertNormalNetwork(n)
    # set all the weights to 1
    ndc._setParameters(ones(12))
    return ndc

if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))

