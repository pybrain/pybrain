__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.utilities import abstractMethod, Named


class RLLearner(Named):
    """ A RL-Learner determines how to change the adaptive parameters of a module,
        but unlike a Trainer (supervised), a RL-learner's dataset has no target but 
        only a reinforcement signal for each sample. It requires access to a 
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
