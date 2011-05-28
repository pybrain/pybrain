"""

Test the forward and backward passes through a linear network.

    >>> from scipy import array
    >>> from pybrain import LinearLayer
    >>> from pybrain.tools.shortcuts import buildNetwork
    >>> n = buildNetwork(2, 4, 3, bias = False, hiddenclass = LinearLayer, recurrent=True)


The forward passes (2 timesteps), by two different but equivalent methods
    >>> input = array([1,2])
    >>> n.inputbuffer[0] = input
    >>> n.forward()
    >>> tmp = n.activate(input * 2)

The backward passes, also by two different but equivalent methods
    >>> outerr = array([-0.1, 0, 1])
    >>> n.outputerror[1] = outerr * 3
    >>> n.backward()
    >>> tmp = n.backActivate(outerr)

Verify that the inputs and outputs are proportional
    >>> sum(n.outputbuffer[1]/n.outputbuffer[0])
    6.0
    >>> abs((n.inputerror[1]/n.inputerror[0])[1] - 3.0) < 0.0001
    True

"""

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.tests import runModuleTestSuite

if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))

