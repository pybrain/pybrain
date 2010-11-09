#!/usr/bin/env python
__author__ = 'Tom Schaul (tom@idsia.ch)'

from pybrain.datasets import SequentialDataSet

class ParityDataSet(SequentialDataSet):
    """ Determine whether the bitstring up to the current point conains a pair number of 1s or not."""
    def __init__(self):
        SequentialDataSet.__init__(self, 1,1)

        self.newSequence()
        self.addSample([-1], [-1])
        self.addSample([1], [1])
        self.addSample([1], [-1])

        self.newSequence()
        self.addSample([1], [1])
        self.addSample([1], [-1])

        self.newSequence()
        self.addSample([1], [1])
        self.addSample([1], [-1])
        self.addSample([1], [1])
        self.addSample([1], [-1])
        self.addSample([1], [1])
        self.addSample([1], [-1])
        self.addSample([1], [1])
        self.addSample([1], [-1])
        self.addSample([1], [1])
        self.addSample([1], [-1])

        self.newSequence()
        self.addSample([1], [1])
        self.addSample([1], [-1])
        self.addSample([-1], [-1])
        self.addSample([-1], [-1])
        self.addSample([-1], [-1])
        self.addSample([-1], [-1])
        self.addSample([-1], [-1])
        self.addSample([-1], [-1])
        self.addSample([-1], [-1])
        self.addSample([1], [1])
        self.addSample([-1], [1])
        self.addSample([-1], [1])
        self.addSample([-1], [1])
        self.addSample([-1], [1])
        self.addSample([-1], [1])

        self.newSequence()
        self.addSample([-1], [-1])
        self.addSample([-1], [-1])
        self.addSample([1], [1])
