__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.learners.learner import Learner


class ValueBasedLearner(Learner):
    """ Learners that learn a value function instead of learning the policy directly. 
    Their policy is then determines such that it maximizes the expected value of the next state. """