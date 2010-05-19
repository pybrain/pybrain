"""
    >>> from pybrain.utilities import flood

The reachable-search can only get to 3 of the points.

    >>> sorted(flood(step, range(10), [2]))
    [2, 4, 5, 7, 8]

Early stopping with relevance argument:

    >>> sorted(flood(step, range(100), [2], relevant=[5]))
    [2, 4, 5]

If the initial point must be included for it to work:

    >>> sorted(flood(step, range(10), [-1]))
    []

"""

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.tests import runModuleTestSuite

def step(x):
    """ A neighbor of x is either 2*x or x+3"""
    return [x+3, 2*x]

if __name__ == '__main__':
    runModuleTestSuite(__import__('__main__'))


