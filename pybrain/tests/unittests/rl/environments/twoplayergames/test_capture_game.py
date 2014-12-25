"""

Initialize a capturegame
    >>> from pybrain.rl.environments.twoplayergames import CaptureGame
    >>> c = CaptureGame(5)
    >>> print(c)
     . . . . .
     . . . . .
     . . . . .
     . . . . .
     . . . . .


Do some moves to produce a situation
    >>> c.performAction([1, (1,0)])
    >>> c.performAction([1, (0,1)])
    >>> c.performAction([1, (1,1)])
    >>> c.performAction([-1, (2,0)])
    >>> c.performAction([-1, (0,2)])
    >>> c.performAction([-1, (1,2)])
    >>> c.performAction([-1, (2,1)])

Now it is almost decided, white has a killing move!
    >>> c.getKilling(-1)
    [(0, 0)]

    >>> c.getWinner()

Do it!
    >>> c.performAction([-1, (0,0)])

White wins.
    >>> c.getWinner()
    -1

Check if all the values are right:
    >>> print(c)
     x X O . .
     X X O . .
     O O . . .
     . . . . .
     . . . . .
    Winner: White (*) (moves done:8)

    >>> correct = {(0, 1): 5, (1, 2): 2, (2, 1): 10, (0, 2): 2, (2, 0): 10, (1, 0): 5, (1, 1): 5}
    >>> c.groups == correct
    True

    >>> correct = {2: set([(0, 3), (1, 3), (2, 2)]), 5: set([(0, 0)]), 10: set([(3, 0), (3, 1), (2, 2)])}
    >>> c.liberties == correct
    True
"""

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.tests import runModuleTestSuite

if __name__ == '__main__':
    runModuleTestSuite(__import__('__main__'))
