"""
    >>> from pybrain.utilities import dictCombinations, subDict, matchingDict
    >>> d1 = {'ones':[1,1,1,'one'], 2:[2,4,6,8], 4:4, 8:['eight']}
    >>> d2 = {1:1, 2:2, 4:4, 8:8}

subDict produces a sub-dictionary, by removing some keys.

    >>> d3 = subDict(d1, ['ones', 2, 4])
    >>> print(sorted(d3.items()))
    [(2, [2, 4, 6, 8]), (4, 4), ('ones', [1, 1, 1, 'one'])]

We can also flip the selection, and limit the keys to the ones NOT in the list:
    >>> d4 = subDict(d1, [8], flip=True)
    >>> d4 == d3
    True


matchingDict determines whether the values of a dictionary match the selection.
No selection always works:

    >>> matchingDict(d2, {})
    True

Not all elements must be present:

    >>> matchingDict(d2, {3:3})
    True

But those that are in both must fit (here 8 is wrong)
    >>> matchingDict(d2, d1)
    False

Without the 8 key:
    >>> matchingDict(d2, d3)
    True

dictCombinations will produce all the combinations of the elements in lists
with their keys, not allowing for identical items,
but dealing with non-lists, and any types of keys and values.

    >>> for x in dictCombinations(d1): print(sorted(x.items()))
    [(2, 2), (4, 4), (8, 'eight'), ('ones', 1)]
    [(2, 4), (4, 4), (8, 'eight'), ('ones', 1)]
    [(2, 6), (4, 4), (8, 'eight'), ('ones', 1)]
    [(2, 8), (4, 4), (8, 'eight'), ('ones', 1)]
    [(2, 2), (4, 4), (8, 'eight'), ('ones', 'one')]
    [(2, 4), (4, 4), (8, 'eight'), ('ones', 'one')]
    [(2, 6), (4, 4), (8, 'eight'), ('ones', 'one')]
    [(2, 8), (4, 4), (8, 'eight'), ('ones', 'one')]


"""

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.tests import runModuleTestSuite

if __name__ == '__main__':
    runModuleTestSuite(__import__('__main__'))
