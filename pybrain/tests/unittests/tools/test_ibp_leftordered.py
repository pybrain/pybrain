"""
    >>> from pybrain.tools.ibp import leftordered
    >>> from scipy import rand, array

Build a random binary matrix

    >>> M = array(rand(10,20)<0.4, dtype=bool)
    >>> L = leftordered(M)

Reordering rows gives the same result

    >>> M2 = M[:, ::-1]
    >>> sum(sum(L == leftordered(M2))) == 200
    True

Reordering columns does not
    >>> M3 = M[::-1, :]
    >>> sum(sum(L == leftordered(M3))) < 200
    True

"""

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.tests import runModuleTestSuite

if __name__ == '__main__':
    runModuleTestSuite(__import__('__main__'))
