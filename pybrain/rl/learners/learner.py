""" The top of the learner hierarchy is more conceptual than functional. 
The different classes distinguish algorithms in such a way that we can automatically 
determine when an algorithm is not applicable for a problem. """

__author__ = 'Tom Schaul, tom@idsia.ch, Thomas Rueckstiess, ruecksti@in.tum.de'


from pybrain.utilities import abstractMethod, Named
from pybrain.datasets import ReinforcementDataSet


# class Learner(object):
#     """ Top-level class for all reinforcement learning algorithms.
#     Any learning algorithm changes a policy (in some way) in order 
#     to increase the expected reward/fitness. """
    
    
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

class Learner(Named):
    """ A Learner determines how to change the adaptive parameters of a module.
        It requires access to a ReinforcementDataSet object (which provides state-action-reward tuples). """

    dataset = None
    module = None

    defaultExploration = None

    def setModule(self, module):
        """ sets the module for the learner. """
        self.module = module    

    def setData(self, rldataset):
        """ sets the dataset for the learner. """
        assert isinstance(rldataset, ReinforcementDataSet)
        self.dataset = rldataset

    def learnOnDataset(self, dataset, *args, **kwargs):
        """ set the dataset, and learn """
        self.setData(dataset)
        self.learnEpisodes(*args, **kwargs)

    def learnEpisodes(self, episodes = 1, *args, **kwargs):
        """ learn on the current dataset, for a number of episodes """
        for dummy in range(episodes):
            self.learn(*args, **kwargs)

    def learn(self):
        """ learn on the current dataset, for a single episode
            @note: has to be implemented by all subclasses. """
        abstractMethod()
