#!/usr/bin/env python
__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.datasets import SupervisedDataSet, ImportanceDataSet


class XORDataSet(SupervisedDataSet):
    """ A dataset for the XOR function."""
    def __init__(self):
        SupervisedDataSet.__init__(self, 2, 1)
        self.addSample([0,0],[0])
        self.addSample([0,1],[1])
        self.addSample([1,0],[1])
        self.addSample([1,1],[0])


class SequentialXORDataSet(ImportanceDataSet):
    """ same thing, but sequential, and having no importance on a second output"""
    def __init__(self):
        ImportanceDataSet.__init__(self, 2, 2)
        self.addSample([0,0],[0, 1],  [1,0])
        self.addSample([0,1],[1, 10], [1,0])
        self.addSample([1,0],[1, -1], [1,0])
        self.addSample([1,1],[0, 0],  [1,0])
