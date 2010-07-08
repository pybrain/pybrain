__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.utilities import abstractMethod


class FitnessEvaluator(object):
    """ The superclass of all evaluators that produce a single output value,
    given an arbitrary input.
    """

    # what would be the desired fitness?
    desiredValue = None

    # what is the desirable direction?
    toBeMinimized = False

    def f(self, x):
        """ The function itself, to be defined by subclasses """
        abstractMethod()

    def __call__(self, x):
        """ All FitnessEvaluators are callable.

        :return: float """
        return self.f(x)
