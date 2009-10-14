__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.rl.learners.learner import Learner

class DirectSearchLearner(Learner):
    """ Meta-class to distinguish direct search learners from other types,
        such as value-based learners.
    """
    pass
