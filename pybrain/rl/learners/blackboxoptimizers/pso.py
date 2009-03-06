#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-

__author__ = ('Julian Togelius, julian@idsia.ch',
              'Justin S Bayer, bayer.justin@googlemail.com')
__version__ = '$Id'


import scipy


from pybrain.rl.learners.blackboxoptimizers.blackboxoptimizer import BlackBoxOptimizer


class Particle(object):
    
    def _setFitness(self, value):
        self._fitness = value
        if value > self.bestFitness:
            self.bestFitness = value
            self.bestPosition = self.position
    
    def _getFitness(self):
        return self._fitness
    
    fitness = property(_getFitness, _setFitness)
    
    def __init__(self, start):
        """Initialize a Particle at the given start vector."""
        self.dim = scipy.size(start)
        self.position = start
        self.velocity = scipy.zeros(scipy.size(start))
        self.bestPosition = scipy.zeros(scipy.size(start))
        self._fitness = None
        self.bestFitness = -scipy.inf
    
    def move(self):
        self.position += self.velocity


def fullyConnected(lst):
    return dict((i, lst) for i in lst)
    
    
def ring(lst):
    leftist = lst[1:] + lst[0:1]
    rightist = lst[-1:] + lst[:-1]
    return dict((i, (j, k)) for i, j, k in zip(lst, leftist, rightist))



class ParticleSwarmOptimizer(BlackBoxOptimizer):
    
    def __init__(self, evaluator, evaluable, size, boundaries=None,
                 memory=2.0, sociality=2.0, inertia=0.9,
                 neighbourfunction=fullyConnected,
                 *args, **kwargs):
        """Initialize a ParticleSwarmOptimizer with `size` particles.
        
        `boundaries` should be a list of (min, max) pairs with the length of the
        dimensionality of the vector to be optimized. Particles will be
        initialized with a position drawn uniformly in that interval.
        
        `memory` indicates how much the velocity of a particle is affected by
        its previous best position.
        `sociality` indicates how much the velocity of a particle is affected by
        its neighbours best position.
        `inertia` is a damping factor.
        """
        super(ParticleSwarmOptimizer, self).__init__(\
            evaluator, evaluable, *args, **kwargs)
        
        self.dim = scipy.size(evaluable)
        self.inertia = inertia
        self.sociality = sociality
        self.memory = memory
        self.neighbourfunction = neighbourfunction
        
        if boundaries is None:
            maxs = scipy.array([10] * self.dim)
            mins = scipy.array([-10] * self.dim)
        else:
            mins = scipy.array([min_ for min_, max_ in boundaries])
            maxs = scipy.array([max_ for min_, max_ in boundaries])
        
        self.particles = []
        for _ in xrange(size):
            startingPosition = scipy.random.random(self.dim)
            startingPosition *= (maxs - mins)
            startingPosition += mins
            self.particles.append(Particle(startingPosition))
        
        # Global neighborhood
        # TODO: do some better neighborhoods later
        self.neighbours = self.neighbourfunction(self.particles)
        
    def best(self, particlelist):
        """Return the particle with the best fitness from a list of particles.
        """
        picker = min if self.minimize else max
        return picker(particlelist, key=lambda p: p.fitness)
    
    def _learnStep(self):
        for particle in self.particles:
            particle.fitness = self.evaluator(particle.position)
            
            # Update the best solutions found so far.
            better = False
            if self.minimize:
                if particle.fitness < self.bestEvaluation:
                    better = True
            else:
                if particle.fitness > self.bestEvaluation:
                    better = True
            if better:
                self.bestEvaluable = particle.position
                self.bestEvaluation = particle.fitness
                
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
            