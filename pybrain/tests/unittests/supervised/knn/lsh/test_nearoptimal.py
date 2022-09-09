# -*- coding: utf-8 -*-

"""
    >>> from scipy import array
    >>> from math import sqrt
    >>> from pybrain.tests import epsilonCheck

Internal Tests:
====================

Let's make a hypercube for 2 dimensions.

    >>> dim = 2

To make some nice sidelengths, we cheat on omega

    >>> omega = 5 / sqrt(sqrt(dim))

    >>> m = nearoptimal.MultiDimHash(dim=dim, omega=omega, prob=0.8)

    >>> m.radius
    1.189207115002...
    >>> m.radiusSquared
    1.41421356237309...

This gives us hypercube sidelength of

    >>> SIDELENGTH = sqrt(sqrt(2)) * omega
    >>> epsilonCheck(SIDELENGTH - 5)
    True

Define some points to work with

    >>> a = array([0, 0])
    >>> m._findHypercube(a)
    (array([0, 0]), array([ 0.,  0.]))

    >>> b = array([0.14 + 3 * SIDELENGTH, .5])
    >>> m._findHypercube(b)
    (array([3, 0]), array([ 0.14,  0.5 ]))

    >>> c = array([.5, 42 * SIDELENGTH + 0.1])
    >>> m._findHypercube(c)
    (array([ 0, 42]), array([ 0.5,  0.1]))

    >>> d = array([-1 * SIDELENGTH + 0.1, 2 * SIDELENGTH + 0.1])
    >>> m._findHypercube(d)
    (array([-1,  2]), array([ 0.1,  0.1]))

Overwrite the balls of the hash to make test the ball intersection function

    >>> m.gridBalls = array([[.3, .3], [ 3., 3.]])

Tests for points within the hypercube [0, 1)^n

    >>> u, v = array([.29, .31]), array([2.9, 2.71])

We discard the first result, since it might trigger a compilation and thus
output some noise.

    >>> _ = m._findLocalBall(u)
    ...

    >>> m._findLocalBall(u)
    0

    >>> m._findLocalBall(v)
    1

Point outside of the hypercube don't return a result

    >>> m._findLocalBall(array([20, 0]))     # Returns None

As do points that are in not within any ball

    >>> m._findLocalBall(array([5.4, .9]))

Testing the composition of _findLocalBall and _findHypercube

    >>> m.findBall(u + array([2 * SIDELENGTH, 4 * SIDELENGTH]))
    ((2, 4), 0)

    >>> m.findBall(u + array([-2 * SIDELENGTH, 4 * SIDELENGTH]))
    ((-2, 4), 0)

    >>> m.findBall(u + array([-2 * SIDELENGTH, -4 * SIDELENGTH]))
    ((-2, -4), 0)

    >>> m.findBall(u + array([2 * SIDELENGTH, -4 * SIDELENGTH]))
    ((2, -4), 0)

    >>> m.findBall(v + array([2 * SIDELENGTH, 4 * SIDELENGTH]))
    ((2, 4), 1)

    >>> m.findBall(v + array([-2 * SIDELENGTH, 4 * SIDELENGTH]))
    ((-2, 4), 1)

    >>> m.findBall(v + array([-2 * SIDELENGTH, -4 * SIDELENGTH]))
    ((-2, -4), 1)

    >>> m.findBall(v + array([2 * SIDELENGTH, -4 * SIDELENGTH]))
    ((2, -4), 1)


Example Usage:
====================

    >>> m = nearoptimal.MultiDimHash(dim=2)
    >>> m.insert(array([0.9585762, 1.15822724]), 'red')
    >>> m.insert(array([1.02331605,  0.95385982]), 'red')
    >>> m.insert(array([0.80838576, 1.07507294]), 'red')
    >>> m.knn(array([0.9585762, 1.15822724]), 1)
    [(array([ 0.9585762 ,  1.15822724]), 'red')]


"""

from pybrain.tests import runModuleTestSuite
from pybrain.supervised.knn.lsh import nearoptimal

if __name__ == "__main__":
   runModuleTestSuite(__import__('__main__'))
