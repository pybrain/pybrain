from __future__ import print_function

__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from numpy import random
from random import sample
from scipy import isscalar

from pybrain.datasets.dataset import DataSet
from pybrain.utilities import fListToString


class SupervisedDataSet(DataSet):
    """SupervisedDataSets have two fields, one for input and one for the target.
    """

    def __init__(self, inp, target):
        """Initialize an empty supervised dataset.

        Pass `inp` and `target` to specify the dimensions of the input and
        target vectors."""
        DataSet.__init__(self)
        if isscalar(inp):
            # add input and target fields and link them
            self.addField('input', inp)
            self.addField('target', target)
        else:
            self.setField('input', inp)
            self.setField('target', target)

        self.linkFields(['input', 'target'])

        # reset the index marker
        self.index = 0

        # the input and target dimensions
        self.indim = self.getDimension('input')
        self.outdim = self.getDimension('target')

    def __reduce__(self):
        _, _, state, _, _ = super(SupervisedDataSet, self).__reduce__()
        creator = self.__class__
        args = self.indim, self.outdim
        return creator, args, state, iter([]), iter({})

    def addSample(self, inp, target):
        """Add a new sample consisting of `input` and `target`."""
        self.appendLinked(inp, target)

    def getSample(self, index=None):
        """Return a sample at `index` or the current sample."""
        return self.getLinked(index)

    def setField(self, label, arr, **kwargs):
        """Set the given array `arr` as the new array of the field specfied by
        `label`."""
        DataSet.setField(self, label, arr, **kwargs)
        # refresh dimensions, in case any of these fields were modified
        if label == 'input':
            self.indim = self.getDimension('input')
        elif label == 'target':
            self.outdim = self.getDimension('target')

    def _provideSequences(self):
        """Return an iterator over sequence lists, although the dataset contains
        only single samples."""
        return iter([[x] for x in iter(self)])

    def evaluateMSE(self, f, **args):
        """Evaluate the predictions of a function on the dataset and return the
        Mean Squared Error, incorporating importance."""
        ponderation = 0.
        totalError = 0
        for seq in self._provideSequences():
            e, p = self._evaluateSequence(f, seq, **args)
            totalError += e
            ponderation += p
        assert ponderation > 0
        return totalError/ponderation

    def _evaluateSequence(self, f, seq, verbose = False):
        """Return the ponderated MSE over one sequence."""
        totalError = 0.
        ponderation = 0.
        for input, target in seq:
            res = f(input)
            e = 0.5 * sum((target-res).flatten()**2)
            totalError += e
            ponderation += len(target)
            if verbose:
                print((    'out:    ', fListToString( list( res ) )))
                print((    'correct:', fListToString( target )))
                print((    'error: % .8f' % e))
        return totalError, ponderation

    def evaluateModuleMSE(self, module, averageOver = 1, **args):
        """Evaluate the predictions of a module on a dataset and return the MSE
        (potentially average over a number of epochs)."""
        res = 0.
        for dummy in range(averageOver):
            module.reset()
            res += self.evaluateMSE(module.activate, **args)
        return res/averageOver

    def splitWithProportion(self, proportion = 0.5, shuffle=True, margin=0):
        """Produce two new datasets, the first one containing the fraction given
        by `proportion` of the samples.

        The first dataset will have a fraction given by `proportion` of the
        dataset chosen randomly from this dataset (using random.permutation).
        The elements in this set will change each time this funciton is called.
        The right (second) dataset will contain the remaining samples, (also permuted randomly).

        Arguments:
            proportion (float): Fraction of dataset to return first in the pair of Datasets returned
                Must be between 0 and 1 inclusive.
                default: 0.5 
            margin (float): Fraction of dataset to be unused when splitting without shuffling.
                This unused portion of the dataset allows the dividing index to shift randomly.
                Must be between 0 and 1 inclusive.
                default: 0  (repeatable nonrandom splits)

        Returns:
            left (Dataset): the portion of the dataset requested of length int(N * portion).
            right (Dataset): the remaining portion of the dataset of length int(N * (1 - portion)).
        """
        separator = int(len(self) * proportion)
        index0, indexN = 0, len(self)
        if shuffle:
            indicies = random.permutation(len(self))
        else:
            indicies = random.np.arange(len(self))
            if margin:
                index_margin = int(margin * len(self))
                index0 = random.randint(0, int(index_margin / 2) + 1)
                indexN = len(self) - index_margin + index0
                assert(indexN <= len(self))
                separator = int((indexN - index0) * proportion)

        leftIndicies = indicies[index0:(index0 + separator)]
        rightIndicies = indicies[(index0 + separator):indexN]

        leftDs = SupervisedDataSet(inp=self['input'][leftIndicies].copy(),
                                   target=self['target'][leftIndicies].copy())
        rightDs = SupervisedDataSet(inp=self['input'][rightIndicies].copy(),
                                    target=self['target'][rightIndicies].copy())
        return leftDs, rightDs


class SequentialSupervisedDataSet(SupervisedDataSet):
    """A SupervisedDataSet is an ordered sequence with two fields, one for input and one for the target

    A SequentialSupervisedDataSet is identical to a SupervisedDataSet except that it maintains
    the order of the samples (both the output and the input). Indices of a new sequence are stored whenever
    the method newSequence() is called. The last (open) sequence is considered
    a normal sequence even though it does not have a following "new sequence"
    marker."""

    def splitWithProportion(self, proportion=0.5, margin=0):
        """Produce two new datasets, each containing a part of the sequences.

        The first dataset will have a fraction given by `proportion` of the
        dataset. This split is repeatable and nonrandom. So the left (first)
        dataset will contain the first M samples unshuffled, where M is int(len(samples) * proportion) 
        and the right (second) dataset will contain the remaining samples, unshuffled.

        Arguments:
            proportion (float): Fraction of dataset to return first in the pair of Datasets returned
                Must be between 0 and 1 inclusive.
                default: 0.5 

        Returns:
            left (Dataset): the portion of the dataset requested of length int(N * portion).
            right (Dataset): the remaining portion of the dataset of length int(N * (1 - portion)).
        """
        return super(SequentialSupervisedDataSet, self).splitWithProportion(proportion=proportion, shuffle=False, margin=margin)

