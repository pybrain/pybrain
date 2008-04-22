__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.utilities import abstractMethod


class Evaluator(object):
    """ The interface for callable objects that return a number, that can be used as evaluators in Learners.
    There are no restrictions on the arguments they might take. 
    They should contain the information if tehy are noisy or deterministic, 
    and (if appropriate) what the highest achievable value is. """
    
    noisy = False
    desiredValue = None
    
    def __call__(self, *args, **kwargs):
        """ @rtype: float """
        abstractMethod()

