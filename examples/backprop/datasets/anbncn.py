__author__ = 'Tom Schaul, tom@idsia.ch and Daan Wierstra'

from pybrain.datasets import SequentialDataSet


class AnBnCnDataSet(SequentialDataSet):
    def __init__(self):
        SequentialDataSet.__init__(self, 0, 1)
        
        self.newSequence()
        self.addSample([],[0])
        self.addSample([],[1])
        self.addSample([],[0])
        self.addSample([],[1])
        self.addSample([],[0])
        self.addSample([],[1])
        
        self.newSequence()
        self.addSample([],[0])
        self.addSample([],[1])
        self.addSample([],[0])
        self.addSample([],[1])
        self.addSample([],[0])
        self.addSample([],[1])
        