__author__ = 'Tom Schaul, tom@idsia.ch'

import copy

from pybrain.utilities import abstractMethod, Named


class Evolvable(Named):
    """ The interface for all Evolvables, i.e. which implement mutation, randomize and copy operators. """

    def mutate(self, **args):
        """ Vary some properties of the underlying module, so that it's behavior
        changes, (but not too abruptly). """
        abstractMethod()

    def copy(self):
        """ By default, returns a full deep-copy - subclasses should implement something faster, if appropriate. """
        return copy.deepcopy(self)

    def randomize(self):
        """ Sets all variable parameters to random values. """
        abstractMethod()

    def newSimilarInstance(self):
        """ Generates a new Evolvable of the same kind."""
        res = self.copy()
        res.randomize()
        return res
