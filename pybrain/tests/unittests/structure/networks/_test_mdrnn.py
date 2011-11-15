"""

    >>> m = _MultiDirectionalMdrnn(2, (4, 4), 1, 1, (2, 2))
    >>> m._permsForSwiping()[0]
    array([0, 1, 2, 3])
    >>> m._permsForSwiping()[1]
    array([1, 0, 3, 2])
    >>> m._permsForSwiping()[2]
    array([2, 3, 0, 1])
    >>> m._permsForSwiping()[3]
    array([3, 2, 1, 0])

    >>> m = _Mdrnn(2, (4, 4), 1, 1, (2, 2))
    >>> m._permsForSwiping()
    [array([0, 1, 2, 3])]
"""


from pybrain.structure.networks.mdrnn import _MultiDirectionalMdrnn, _Mdrnn #@UnusedImport
from pybrain.tests import runModuleTestSuite


if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))
