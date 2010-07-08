""" The top of the learner hierarchy is more conceptual than functional.
The different classes distinguish algorithms in such a way that we can automatically
determine when an algorithm is not applicable for a problem. """

__author__ = 'Tom Schaul, tom@idsia.ch, Thomas Rueckstiess, ruecksti@in.tum.de'


from pybrain.utilities import abstractMethod
import logging


class Learner(object):
    """ Top-level class for all reinforcement learning algorithms.
    Any learning algorithm changes a policy (in some way) in order
    to increase the expected reward/fitness.
    """

    module = None

    def learn(self):
        """ The main method, that invokes a learning step. """
        abstractMethod()


class ExploringLearner(Learner):
    """ A Learner determines how to change the adaptive parameters of a module.
    """

    explorer = None

    def explore(self, state, action):
        if self.explorer is not None:
            return self.explorer.activate(state, action)
        else:
            # logging.warning("No explorer found: no exploration could be done.")
            return action


class EpisodicLearner(Learner):
    """ Assumes the task is episodic, not life-long,
    and therefore does a learning step only after the end of each episode. """

    def learnEpisodes(self, episodes = 1, *args, **kwargs):
        """ learn on the current dataset, for a number of episodes """
        for _ in range(episodes):
            self.learn(*args, **kwargs)

    def newEpisode(self):
        """ informs the learner that a new episode has started. """
        if self.explorer is not None:
            self.explorer.newEpisode()

    def reset(self):
        pass


class DataSetLearner(EpisodicLearner):
    """ A class for learners that learn from a dataset, which has no target output but
        only a reinforcement signal for each sample. It requires a
        ReinforcementDataSet object (which provides state-action-reward tuples). """

    dataset = None

    def learnOnDataset(self, dataset, *args, **kwargs):
        """ set the dataset, and learn """
        self.dataset = dataset
        self.learnEpisodes(*args, **kwargs)

