"""

    >>> from pybrain.structure.modules.samplelayer import BernoulliLayer
    >>> from scipy import random, array, empty

Set the random seed so we can predict the random variables.

    >>> random.seed(0)

Create a layer.

    >>> layer = BernoulliLayer(3)
    >>> input = array((0.8, 0.5, 0.2))
    >>> output = empty((3,))

Now test some forwards:

    >>> layer._forwardImplementation(input, output)
    >>> output
    array([ 0.,  1.,  1.])

    >>> layer._forwardImplementation(input, output)
    >>> output
    array([ 0.,  0.,  1.])

    >>> layer._forwardImplementation(input, output)
    >>> output
    array([ 0.,  1.,  1.])

"""

__author__ = 'Justin Bayer, bayerj@in.tum.de'

from pybrain.tests import runModuleTestSuite

if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))

