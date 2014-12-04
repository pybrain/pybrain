"""

Initialize a game of Pente.
    >>> from pybrain.rl.environments.twoplayergames.pente import PenteGame
    >>> dim = 5
    >>> c = PenteGame((dim, dim))
    >>> print(c)
     _ _ _ _ _
     _ _ _ _ _
     _ _ * _ _
     _ _ _ _ _
     _ _ _ _ _
    Black captured:0, white captured:0.


Do some moves to produce a situation
    >>> c.performAction([1, (0,1)])
    >>> c.performAction([-1, (1,0)])
    >>> c.performAction([1, (1,1)])
    >>> c.performAction([-1, (1,2)])
    >>> c.performAction([1, (2,0)])
    >>> c.performAction([-1, (2,1)])
    >>> c.performAction([1, (0,2)])

Show the updated board:
    >>> print(c)
     _ # # _ _
     * # * _ _
     # * * _ _
     _ _ _ _ _
     _ _ _ _ _
    Black captured:0, white captured:0.


Do some captures:
    >>> c.performAction([-1, (0,3)])
    >>> c.performAction([1, (3,2)])
    >>> c.performAction([-1, (0,0)])

Stepping between black stones is not deadly though:
    >>> c.performAction([1, (2,3)])
    >>> c.performAction([-1, (2,2)])
    >>> print(c)
     * _ _ * _
     * # _ _ _
     # * * # _
     _ _ # _ _
     _ _ _ _ _
    Black captured:1, white captured:1.

Fast forward to the end:
    >>> c.performAction([-1, (0,4)])
    >>> c.performAction([-1, (3,1)])
    >>> c.performAction([-1, (4,0)])

Now it is almost decided, white has a killing move!
    >>> c.getKilling(-1)
    [(1, 3)]

    >>> c.getWinner()

Do it!
    >>> c.performAction([-1, (1,3)])

White wins.
    >>> c.getWinner()
    -1

Check if all the values are right:
    >>> print(c)
     * _ _ * *
     * # _ x _
     # * * # _
     _ * # _ _
     * _ _ _ _
    Winner: White (*) (moves done:17)
    Black captured:1, white captured:1.


"""

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.tests import runModuleTestSuite

if __name__ == '__main__':
    runModuleTestSuite(__import__('__main__'))


