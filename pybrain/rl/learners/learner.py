""" The top of the learner hierarchy is more conceptual than functional. 
The different classes distinguish algorithms in such a way that we can automatically 
determine when an algorithm is not applicable for a problem. """

__author__ = 'Tom Schaul, tom@idsia.ch'


class Learner(object):
    """ Top-level class for all reinforcement learning algorithms.
    Any learning algorithm changes a policy (in some way) in order 
    to increase the expected reward/fitness. """
    
    
class OntogeneticLearner(Learner):
    """ The class of classical RL algorithms. 
    They make use of observations, actions and rewards. """
    
    
class OffPolicyLearner(Learner):
    """ A kind of learner that can learn from offline data,
    gathered independently of the current policy (usually at an earlier time).
    
    All off-policy learners can also learn on-policy. """
    
    
class EpisodicLearner(Learner):
    """ Assumes the task is episodic, not life-long,
    and therefore does a learning step only after the end of each episode. """
    
    
class PhylogeneticLearner(EpisodicLearner):
    """ The opposite of an ontogenetic algorithm. 
    It makes use of only the cumulative reward (=fitness) at the end of the episode.
    """
    
