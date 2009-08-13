__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.utilities import abstractMethod
from pybrain.rl.learners.learner import Learner


class DataSetLearner(Learner):
    """ A class for learners that learn from a dataset, which has no target output but 
        only a reinforcement signal for each sample. It requires a 
        ReinforcementDataSet object (which provides state-action-reward tuples). """

    ds = None
    module = None

    def setModule(self, module):
        self.module = module    

    def setData(self, rldataset):
        # @todo assert that rldataset is of class ReinforcementDataSet
        self.ds = rldataset

    def learnOnDataset(self, dataset, *args, **kwargs):
        """ set the dataset, and learn """
        self.setData(dataset)
        self.learnEpochs(*args, **kwargs)

    def learnEpochs(self, epochs = 1, *args, **kwargs):
        """ learn on the current dataset, for a number of epochs """
        for dummy in range(epochs):
            self.learn(*args, **kwargs)

    def learn(self):
        """ learn on the current dataset, for a single epoch
            @note: has to be implemented by all subclasses. """
        abstractMethod()
