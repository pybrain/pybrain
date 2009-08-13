__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.learners.learner import Learner


class DirectSearch(Learner):
    """ Learning algorithm that operates directly on policies, without trying to learn 
    a value-function for each (believed) state. """
    