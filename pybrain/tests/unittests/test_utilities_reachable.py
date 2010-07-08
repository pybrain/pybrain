"""
    >>> from pybrain.utilities import reachable, decrementAny, flood

The reachable-search can only get to 3 of the points.

    >>> dests = [(1,3), (2,2), (3,2), (3,1), (1,0), (0,2), (2,0), (0,1)]
    >>> sorted(reachable(decrementAny, [(3,3)], dests).items())
    [((1, 3), 2), ((2, 2), 2), ((3, 2), 1)]

    >>> d2 = map(lambda i: (i,), range(10))
    >>> reachable(decrementAny, [(12,), (29,)], d2)
    {(9,): 3}

"""

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.tests import runModuleTestSuite

if __name__ == '__main__':
    runModuleTestSuite(__import__('__main__'))
