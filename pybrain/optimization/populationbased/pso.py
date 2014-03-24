__author__ = ('Julian Togelius, julian@idsia.ch',
              'Justin S Bayer, bayer.justin@googlemail.com')

import scipy
import logging

from pybrain.optimization.optimizer import ContinuousOptimizer


def fullyConnected(lst):
    return dict((i, lst) for i in lst)

def ring(lst):
    leftist = lst[1:] + lst[0:1]
    rightist = lst[-1:] + lst[:-1]
    return dict((i, (j, k)) for i, j, k in zip(lst, leftist, rightist))

# TODO: implement some better neighborhoods


class ParticleSwarmOptimizer(ContinuousOptimizer):
    """ Particle Swarm Optimization

    `size` determines the number of particles.

    `boundaries` should be a list of (min, max) pairs with the length of the
    dimensionality of the vector to be optimized (default: +-10). Particles will be
    initialized with a position drawn uniformly in that interval.

    `memory` indicates how much the velocity of a particle is affected by
    its previous best position.

    `sociality` indicates how much the velocity of a particle is affected by
    its neighbours best position.

    `inertia` is a damping factor.
    """

    size = 20
    boundaries = None

    memory = 2.0
    sociality = 2.0
    inertia = 0.9

    neighbourfunction = None

    mustMaximize = True

    def _setInitEvaluable(self, evaluable):
        if evaluable is not None:
            logging.warning("Initial point provided was ignored.")
        ContinuousOptimizer._setInitEvaluable(self, evaluable)

    def _additionalInit(self):
        self.dim = self.numParameters
        if self.neighbourfunction is None:
            self.neighbourfunction = fullyConnected

        if self.boundaries is None:
            maxs = scipy.array([10] * self.dim)
            mins = scipy.array([-10] * self.dim)
        else:
            mins = scipy.array([min_ for min_, max_ in self.boundaries])
            maxs = scipy.array([max_ for min_, max_ in self.boundaries])

        self.particles = []
        for _ in range(self.size):
            startingPosition = scipy.random.random(self.dim)
            startingPosition *= (maxs - mins)
            startingPosition += mins
            self.particles.append(Particle(startingPosition, self.minimize))

        # Global neighborhood
        self.neighbours = self.neighbourfunction(self.particles)

    def best(self, particlelist):
        """Return the particle with the best fitness from a list of particles.
        """
        picker = min if self.minimize else max
        return picker(particlelist, key=lambda p: p.fitness)

    def _learnStep(self):
        for particle in self.particles:
            particle.fitness = self._oneEvaluation(particle.position.copy())

        for particle in self.particles:
            bestPosition = self.best(self.neighbours[particle]).position
            diff_social = self.sociality \
                          * scipy.random.random() \
                          * (bestPosition - particle.position)

            diff_memory = self.memory \
                          * scipy.random.random() \
                          * (particle.bestPosition - particle.position)

            particle.velocity *= self.inertia
            particle.velocity += diff_memory + diff_social
            particle.move()

    @property
    def batchSize(self):
        return self.size


class Particle(object):
    def __init__(self, start, minimize):
        """Initialize a Particle at the given start vector."""
        self.minimize = minimize
        self.dim = scipy.size(start)
        self.position = start
        self.velocity = scipy.zeros(scipy.size(start))
        self.bestPosition = scipy.zeros(scipy.size(start))
        self._fitness = None
        if self.minimize:
            self.bestFitness = scipy.inf
        else:
            self.bestFitness = -scipy.inf

    def _setFitness(self, value):
        self._fitness = value
        if ((self.minimize and value < self.bestFitness)
            or (not self.minimize and value > self.bestFitness)):
            self.bestFitness = value
            self.bestPosition = self.position.copy()

    def _getFitness(self):
        return self._fitness

    fitness = property(_getFitness, _setFitness)

    def move(self):
        self.position += self.velocity

