"""


    >>> from pybrain.optimization.populationbased.pso import ring
    >>> ring(list(range(9)))
    {0: (1, 8), 1: (2, 0), 2: (3, 1), 3: (4, 2), 4: (5, 3), 5: (6, 4), 6: (7, 5), 7: (8, 6), 8: (0, 7)}

    Disabled:
    lattice(list(range(9)))
    {0: (1, 2),
     1: (0, 3),
     2: (0, 3),
     3: (1, 2)
     4:
     5:
     6:
     7:
     8:
     9:}

"""

__author__ = ('Justin Bayer, bayer.justin@googlemail.com;'
              'Julian Togelius, julian@idsia.ch')

from pybrain.tests import runModuleTestSuite

if __name__ == '__main__':
    runModuleTestSuite(__import__('__main__'))


