#!/usr/bin/env python
__author__ = 'Tom Schaul, tom@idsia.ch and Daan Wierstra'

from pybrain.datasets import SequentialDataSet

# TODO: make it *real* AnBnCn

class AnBnCnDataSet(SequentialDataSet):
    """ A Dataset partially modeling an AnBnCn grammar. """

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
