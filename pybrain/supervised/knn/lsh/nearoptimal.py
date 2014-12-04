

"""This module provides functionality for locality sensitive hashing in high
dimensional euclidean spaces.

It is based on the work of Andoni and Indyk, 'Near-Optimal Hashing Algorithms
for Approximate Nearest Neighbor in High Dimensions'."""


__author__ = 'Justin Bayer, bayer.justin@googlemail.com'


import logging

from collections import defaultdict
from heapq import nlargest
from math import sqrt, log, ceil

from scipy import array, dot, random, ones

try:
    # Python 2
    from scipy import weave
except ImportError:
    # Python 3
    pass


class MultiDimHash(object):
    """Class that represents a datastructure that enables nearest neighbours
    search and methods to do so."""

    # If the dimension of a dataset is bigger than this bound, the
    # dimensionality will be reduced by a random projection into 24dimensional
    # space
    lowerDimensionBound = 24

    def _getRadius(self):
        return self._radius

    def _setRadius(self, value):
        self._radius = abs(value)
        self.radiusSquared = value ** 2

    radius = property(_getRadius, _setRadius)

    def __init__(self, dim, omega=4, prob=0.8):
        """Create a hash for arrays of dimension dim.

        The hyperspace will be split into hypercubes with a sidelength of
        omega * sqrt(sqrt(dim)), that is omega * radius.

        Every point in the dim-dimensional euclidean space will be hashed to
        its correct bucket with a probability of prob.

        """
        message = ("Creating Hash with %i dimensions, sidelength %.2f and " +
                  "cNN-probability %.2f") % (dim, omega, prob)
        logging.debug(message)

        self.dim = dim
        self.omega = omega
        self.prob = prob

        self.radius = sqrt(sqrt(min(dim, self.lowerDimensionBound)))
        logging.debug("Radius set to %.2f" % self.radius)

        self._initializeGrids()
        self._initializeProjection()

        self.balls = defaultdict(lambda: [])

    def _findAmountOfGrids(self):
        w = self.radius
        omega = self.omega
        d = self.dim
        prob = self.prob

        N = ((omega * w) / (w / sqrt(d))) ** d
        result = int(ceil(log((1 - prob) / N, 1 - 1 / N)))
        logging.debug("Number of grids: %i" % result)
        return result

    def _initializeGrids(self):
        offset = self.omega * self.radius
        radius_offset = ones(self.dim) * self.radius
        self.gridBalls = random.random((self._findAmountOfGrids(), self.dim))
        self.gridBalls *= offset
        self.gridBalls += radius_offset

    def _initializeProjection(self):
        if self.dim <= self.lowerDimensionBound:
            # We only need to reduce the dimension if it's bigger than
            # lowerDimensionBound; otherwise, chose identity
            self.projection = 1
        else:
            projection_shape = self.dim, self.lowerDimensionBound
            self.projection = random.standard_normal(projection_shape)
            self.projection /= sqrt(self.lowerDimensionBound)

    def _findHypercube(self, point):
        """Return where a point lies in what hypercube.

        The result is a pair of two arrays. The first array is an array of
        integers that indicate the multidimensional index of the hypercube it
        is in. The second array is an array of floats, specifying the
        coordinates of the point in that hypercube.
        """
        offset = self.omega * self.radius
        divmods = (divmod(p, offset) for p in point)
        hypercube_indices, relative_point = [], []
        for index, rest in divmods:
            hypercube_indices.append(index)
            relative_point.append(rest)
        return array(hypercube_indices, dtype=int), array(relative_point)

    def _findLocalBall_noinline(self, point):
        """Return the index of the ball that the point lies in."""
        for i, ball in enumerate(self.gridBalls):
            distance = point - ball
            if dot(distance.T, distance) <= self.radiusSquared:
                return i

    def _findLocalBall_inline(self, point):
        """Return the index of the ball that the point lies in."""
        balls = self.gridBalls
        nBalls, dim = balls.shape #@UnusedVariable
        radiusSquared = self.radiusSquared #@UnusedVariable

        code = """
            #line 121 "nearoptimal.py"
            return_val = -1;
            for (long i = 0; i < nBalls; i++)
            {
                double distance = 0.0;
                for (long j = 0; j < dim; j++)
                {
                    double diff = balls(i, j) - point(j);
                    distance += diff * diff;
                }
                if (distance <= radiusSquared) {
                    return_val = i;
                    break;
                }
            }
        """

        variables = 'point', 'balls', 'nBalls', 'dim', 'radiusSquared',
        result = weave.inline(
            code,
            variables,
            type_converters=weave.converters.blitz,
            compiler='gcc')

        return result if result != -1 else None

    _findLocalBall = _findLocalBall_noinline

    def findBall(self, point):
        hypercube_index, relative_point = self._findHypercube(point)
        ball_index = self._findLocalBall(relative_point)
        return tuple(hypercube_index), ball_index

    def insert(self, point, satellite):
        """Put a point and its satellite information into the hash structure.
        """
        point = dot(self.projection, point)
        index = self.findBall(point)
        self.balls[index].append((point, satellite))

    def _findKnnCandidates(self, point):
        """Return a set of candidates that might be nearest neighbours of a
        query point."""
        index = self.findBall(point)
        logging.debug("Found %i candidates for cNN" % len(self.balls[index]))
        return self.balls[index]

    def knn(self, point, k):
        """Return the k approximate nearest neighbours of the item in the
        current hash.

        Mind that the probabilistic nature of the data structure might not
        return a nearest neighbor at all and not the nearest neighbour."""

        candidates = self._findKnnCandidates(point)

        def sortKey(xxx_todo_changeme):
            (point_, satellite_) = xxx_todo_changeme
            distance = point - point_
            return - dot(distance.T, distance)

        return nlargest(k, candidates, key=sortKey)
