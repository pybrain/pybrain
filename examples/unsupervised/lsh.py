#!/usr/bin/env python


__author__ = 'Justin Bayer, bayer.justin@googlemail.com'


import logging
from random import shuffle

from pylab import show, plot, clf
from pybrain.supervised.knn.lsh.nearoptimal import MultiDimHash
from scipy import random, array, dot, zeros
from scipy.linalg import orth


def randomRotation(dim):
    """Return a random rotation matrix of rank dim."""
    return orth(random.random((dim, dim)))


def makeData(amount = 10000):
    """Return 2D dataset of points in (0, 1) where points in a circle of
    radius .4 around the center are blue and all the others are red."""
    center = array([0.5, 0.5])

    def makePoint():
        """Return a random point and its satellite information.

        Satellite is 'blue' if point is in the circle, else 'red'."""
        point = random.random((2,)) * 10
        vectorLength = lambda x: dot(x.T, x)
        return point, 'blue' if vectorLength(point - center) < 25 else 'red'

    return [makePoint() for _ in range(amount)]


if __name__ == '__main__':
    # Amount of dimensions to test with
    dimensions = 3

    loglevel = logging.DEBUG
    logging.basicConfig(level=loglevel,
                        format='%(asctime)s %(levelname)s %(message)s')

    logging.info("Making dataset...")
    data = makeData(1000)

    logging.info("Making random projection...")
    proj = zeros((2, dimensions))
    proj[0, 0] = 1
    proj[1, 1] = 1
    randRot = randomRotation(dimensions)
    proj = dot(proj, randRot)

    logging.info("Initializing data structure...")
    m = MultiDimHash(dimensions, 2, 0.80)

    logging.info("Putting data into hash...")
    for point, satellite in data:
        point = dot(point, proj)
        m.insert(point, satellite)

    logging.info("Retrieve nearest neighbours...")
    result = []
    width, height = 2**5, 2**5
    grid = (array([i / width * 10, j / height * 10])
            for i in range(width)
            for j in range(height))
    projected_grid = [(p, dot(p, proj)) for p in grid]

    # Just to fake random access
    shuffle(projected_grid)

    for p, pp in projected_grid:
        nns = m.knn(pp, 1)
        if nns == []:
            continue
        _, color = nns[0]
        result.append((p, color))

    # Visualize it
    visualize = True
    if visualize:
        clf()
        result = [((x, y), color)
                  for (x, y), color in result
                  if color is not None]

        xs_red = [x for ((x, y), color) in result if color == 'red']
        ys_red = [y for ((x, y), color) in result if color == 'red']
        xs_blue = [x for ((x, y), color) in result if color == 'blue']
        ys_blue = [y for ((x, y), color) in result if color == 'blue']

        plot(xs_red, ys_red, 'ro')
        plot(xs_blue, ys_blue, 'bo')
        show()

    ballsizes = (len(ball) for ball in m.balls.values())
    logging.info("Sizes of the balls: " + " ".join(str(i) for i in ballsizes))

    logging.info("Finished")
