__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'
# $Id$

from scipy import ravel, r_
from random import sample

from pybrain.datasets.supervised import SupervisedDataSet


class EmptySequenceError(Exception): pass


class SequentialDataSet(SupervisedDataSet):
    """A SequentialDataSet is like a SupervisedDataSet except that it can keep
    track of sequences of samples. Indices of a new sequence are stored whenever
    the method newSequence() is called. The last (open) sequence is considered
    a normal sequence even though it does not have a following "new sequence"
    marker."""

    def __init__(self, indim, targetdim):
        SupervisedDataSet.__init__(self, indim, targetdim)
        # add field that stores the beginning of a new episode
        self.addField('sequence_index', 1)
        self.append('sequence_index', 0)
        self.currentSeq = 0

    def newSequence(self):
        """Marks the beginning of a new sequence. this function does nothing if
        called at the very start of the data set. Otherwise, it starts a new
        sequence. Empty sequences are not allowed, and an EmptySequenceError
        exception will be raised."""
        length = self.getLength()
        if length != 0:
            if ravel(self.getField('sequence_index'))[-1] == length:
                raise EmptySequenceError
            self._appendUnlinked('sequence_index', length)

    def _getSequenceField(self, index, field):
        """Return a sequence of one single field given by `field` and indexed by
        `index`."""
        seq = ravel(self.getField('sequence_index'))
        if len(seq) == index + 1:
            # user wants to access the last sequence, return until end of data
            return self.getField(field)[ravel(self.getField('sequence_index'))[index]:]
        if len(seq) < index + 1:
            # sequence index beyond number of sequences. raise exception
            raise IndexError('sequence does not exist.')
        return self.getField(field)[ravel(self.getField('sequence_index'))[index]:ravel(self.getField('sequence_index'))[index + 1]]

    def getSequence(self, index):
        """Returns the sequence given by `index`.

        A list of arrays is returned for the linked arrays. It is assumed that
        the last sequence goes until the end of the dataset."""
        return [self._getSequenceField(index, l) for l in self.link]

    def getSequenceIterator(self, index):
        """Return an iterator over the samples of the sequence specified by
        `index`."""
        fields = self.getSequence(index)
        for i in range(self.getSequenceLength(index)):
            yield [f[i] for f in fields]

    def endOfSequence(self, index):
        """Return True if the marker was moved over the last element of
        sequence `index`, False otherwise.

        Mostly used like .endOfData() with while loops."""
        seq = ravel(self.getField('sequence_index'))
        if len(seq) == index + 1:
            # user wants to access the last sequence, return until end of data
            return self.endOfData()
        if len(seq) < index + 1:
            # sequence index beyond number of sequences. raise exception
            raise IndexError('sequence does not exist.')
        else:
            return self.index >= seq[index + 1]

    def gotoSequence(self, index):
        """Move the internal marker to the beginning of sequence `index`."""
        try:
            self.index = ravel(self.getField('sequence_index'))[index]
        except IndexError:
            raise IndexError('sequence does not exist')

    def getCurrentSequence(self):
        """Return the current sequence, according to the marker position."""
        seq = ravel(self.getField('sequence_index'))
        return len(seq) - sum(seq > self.index) - 1

    def getNumSequences(self):
        """Return the number of sequences. The last (open) sequence is also
        counted in, even though there is no additional 'newSequence' marker."""
        return self.getField('sequence_index').shape[0]

    def getSequenceLength(self, index):
        """Return the length of the given sequence. If `index` is pointing
        to the last sequence, the sequence is considered to go until the end
        of the dataset."""
        seq = ravel(self.getField('sequence_index'))
        if len(seq) == index + 1:
            # user wants to access the last sequence, return until end of data
            return int(self.getLength() - seq[index])
        if len(seq) < index + 1:
            # sequence index beyond number of sequences. raise exception
            raise IndexError('sequence does not exist.')
        return int(seq[index + 1] - seq[index])

    def removeSequence(self, index):
        """Remove the `index`'th sequence from the dataset and places the
        marker to the sample following the removed sequence."""
        if index >= self.getNumSequences():
            # sequence doesn't exist, raise exception
            raise IndexError('sequence does not exist.')
        sequences = ravel(self.getField('sequence_index'))
        seqstart = sequences[index]
        if index == self.getNumSequences() - 1:
            # last sequence is going to be removed
            lastSeqDeleted = True
            seqend = self.getLength()
        else:
            lastSeqDeleted = False
            # sequence to remove is not last one (sequence_index exists)
            seqend = sequences[index + 1]

        # cut out data from all fields
        for label in self.link:
            # concatenate rows from start to seqstart and from seqend to end
            self.data[label] = r_[self.data[label][:seqstart, :], self.data[label][seqend:, :]]
            # update endmarkers of linked fields
            self.endmarker[label] -= seqend - seqstart

        # update sequence indices
        for i, val in enumerate(sequences):
            if val > seqstart:
                self.data['sequence_index'][i, :] -= seqend - seqstart

        # remove sequence index of deleted sequence and reduce its endmarker
        self.data['sequence_index'] = r_[self.data['sequence_index'][:index, :], self.data['sequence_index'][index + 1:, :]]
        self.endmarker['sequence_index'] -= 1

        if lastSeqDeleted:
            # last sequence was removed
            # move sequence marker to last remaining sequence
            self.currentSeq = index - 1
            # move sample marker to end of dataset
            self.index = self.getLength()
            # if there was only 1 sequence left, re-initialize sequence index
            if self.getLength() == 0:
                self.clear()
        else:
            # removed sequence was not last one (sequence_index exists)
            # move sequence marker to the new sequence at position 'index'
            self.currentSeq = index
            # move sample marker to beginning of sequence at position 'index'
            self.index = ravel(self.getField('sequence_index'))[index]


    def clear(self):
        SupervisedDataSet.clear(self, True)
        self._appendUnlinked('sequence_index', [0])
        self.currentSeq = 0

    def __iter__(self):
        """Create an iterator object over sequences which are themselves
        iterable objects."""
        for i in range(self.getNumSequences()):
            yield self.getSequenceIterator(i)

    def _provideSequences(self):
        """Return an iterator over sequence lists."""
        return iter(map(list, iter(self)))

    def evaluateModuleMSE(self, module, averageOver=1, **args):
        """Evaluate the predictions of a module on a sequential dataset
        and return the MSE (potentially average over a number of epochs)."""
        res = 0.
        for dummy in range(averageOver):
            ponderation = 0.
            totalError = 0
            for seq in self._provideSequences():
                module.reset()
                e, p = self._evaluateSequence(module.activate, seq, **args)
                totalError += e
                ponderation += p
            assert ponderation > 0
            res += totalError / ponderation
        return res / averageOver

    def splitWithProportion(self, proportion=0.5):
        """Produce two new datasets, each containing a part of the sequences.

        The first dataset will have a fraction given by `proportion` of the
        dataset."""
        l = self.getNumSequences()
        leftIndices = sample(range(l), int(l * proportion))
        leftDs = self.copy()
        leftDs.clear()
        rightDs = leftDs.copy()
        index = 0
        for seq in iter(self):
            if index in leftIndices:
                leftDs.newSequence()
                for sp in seq:
                    leftDs.addSample(*sp)
            else:
                rightDs.newSequence()
                for sp in seq:
                    rightDs.addSample(*sp)
            index += 1
        return leftDs, rightDs

