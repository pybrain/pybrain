"""
Build a CaptureGameNetwork with LSTM cells
    >>> from pybrain.structure.networks.custom import CaptureGameNetwork
    >>> from pybrain import MDLSTMLayer
    >>> size = 2
    >>> n = CaptureGameNetwork(size = size, componentclass = MDLSTMLayer, hsize = 1, peepholes = False)

Check it's string representation
    >>> print(n)
    CaptureGameNetwork-s2-h1-MDLSTMLayer--...
      Modules:
        [<BiasUnit 'bias'>, <LinearLayer 'input'>, <MDLSTMLayer 'hidden(0, 0, 0)'>, ... <MDLSTMLayer 'hidden(0, 0, 3)'>, <SigmoidLayer 'output'>]
      Connections:
        [<IdentityConnection ...


Check some of the connections dimensionalities
    >>> c1 = n.connections[n['hidden(1, 0, 3)']][0]
    >>> c2 = n.connections[n['hidden(0, 1, 2)']][-1]
    >>> print((c1.indim, c1.outdim))
    (1, 1)
    >>> print((c2.indim, c2.outdim))
    (1, 1)
    >>> n.paramdim
    21

Try writing it to an xml file, reread it and determine if it looks the same:

    >>> from pybrain.tests import xmlInvariance
    >>> xmlInvariance(n)
    Same representation
    Same function
    Same class

Check its gradient:

    >>> from pybrain.tests import gradientCheck
    >>> gradientCheck(n)
    Perfect gradient
    True

"""

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.tests import runModuleTestSuite

if __name__ == '__main__':
    runModuleTestSuite(__import__('__main__'))
