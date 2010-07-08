"""
    >>> from pybrain.utilities import avgFoundAfter

A sequence of decreasing target values
    >>> dess = [20,10,3,1,0]

A list of sequences of encountered values
    >>> ls = [[11,11,11,11,11,11,11,11,1,1,1,10,1,0],\
              [11,9,7,5,2,0.5,-2],\
              [2,2,2,2,2,0,2,2,0,2,2,2,-1]]

Average index where each value is encountered.
    >>> avgFoundAfter(dess, ls)
    array([ 0.,  3.,  4.,  6.,  8.])

If a value is not always encountered, the length of the longest sequence is used:
    >>> avgFoundAfter([10,0], [[20],[20,1,1,1,-2]])
    array([ 3. ,  4.5])


"""

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.tests import runModuleTestSuite

if __name__ == '__main__':
    runModuleTestSuite(__import__('__main__'))



