__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

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
        return iter(map(lambda x: [x], iter(self)))

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
                print(    'out:    ', fListToString( list( res ) ))
                print(    'correct:', fListToString( target ))
                print(    'error: % .8f' % e)
        return totalError, ponderation

    def evaluateModuleMSE(self, module, averageOver = 1, **args):
        """Evaluate the predictions of a module on a dataset and return the MSE
        (potentially average over a number of epochs)."""
        res = 0.
        for dummy in range(averageOver):
            module.reset()
            res += self.evaluateMSE(module.activate, **args)
        return res/averageOver

    def splitWithProportion(self, proportion = 0.5):
        """Produce two new datasets, the first one containing the fraction given
        by `proportion` of the samples."""
        leftIndices = set(sample(range(len(self)), int(len(self)*proportion)))
        leftDs = self.copy()
        leftDs.clear()
        rightDs = leftDs.copy()
        index = 0
        for sp in self:
            if index in leftIndices:
                leftDs.addSample(*sp)
            else:
                rightDs.addSample(*sp)
            index += 1
        return leftDs, rightDs

