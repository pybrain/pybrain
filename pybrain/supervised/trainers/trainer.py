__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.utilities import Named, abstractMethod


class Trainer(Named):
    """ A trainer determines how to change the adaptive parameters of a module. 
    It requires access to a DataSet object (which provides input-target tuples). """
    # e.g. bptt, rtrl, svm
    
    ds = None
    module = None
    
    def __init__(self, module):
        self.module = module    
    
    def setData(self, dataset):
        self.ds = dataset
        if dataset:
            assert dataset.indim == self.module.indim
            assert dataset.outdim == self.module.outdim
        
    def trainOnDataset(self, dataset, *args, **kwargs):
        """ set the dataset, and train """
        self.setData(dataset)
        self.trainEpochs(*args, **kwargs)
        
    def trainEpochs(self, epochs = 1, *args, **kwargs):
        """ train on the current dataset, for a number of epochs """
        for dummy in range(epochs):
            self.train(*args, **kwargs)
        
    def train(self):
        """ train on the current dataset, for a single epoch
            @note: has to be implemented by all subclasses. """
        abstractMethod()

        