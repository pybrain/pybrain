from __future__ import with_statement

__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

import random
import pickle
from itertools import chain
from scipy import zeros, resize, ravel, asarray
import scipy

from pybrain.utilities import Serializable


class OutOfSyncError(Exception): pass
class VectorFormatError(Exception): pass
class NoLinkedFieldsError(Exception): pass


class DataSet(Serializable):
    """DataSet is a general base class for other data set classes
    (e.g. SupervisedDataSet, SequentialDataSet, ...). It consists of several
    fields. A field is a NumPy array with a label (a string) attached to it.
    Fields can be linked together which means they must have the same length."""

    def __init__(self):
        self.data = {}
        self.endmarker = {}
        self.link = []
        self.index = 0

        # row vectors returned by getLinked can have different formats:
        # '1d'       example: array([1, 2, 3])
        # '2d'       example: array([[1, 2, 3]])
        # 'list'     example: [1, 2, 3]
        self.vectorformat = 'none'

    def __str__(self):
        """Return a string representation of a dataset."""
        s = ""
        for key in self.data:
            s = s + key + ": dim" + str(self.data[key].shape) + "\n" + str(self.data[key][:self.endmarker[key]]) + "\n\n"
        return s

    def __getitem__(self, field):
        """Return the given field."""
        return self.getField(field)

    def __iter__(self):
        self.reset()
        while not self.endOfData():
            yield self.getLinked()

    def getVectorFormat(self):
        """Returns the current vector format."""
        return self.__vectorformat

    def setVectorFormat(self, vf):
        """Determine which format to use for returning vectors. Use the property vectorformat.

            :key type: possible types are '1d', '2d', 'list'
                  '1d' - example: array([1,2,3])
                  '2d' - example: array([[1,2,3]])
                'list' - example: [1,2,3]
                'none' - no conversion
         """
        switch = {
            '1d': self._convertArray1d,
            '2d': self._convertArray2d,
            'list': self._convertList,
            'none': lambda x:x
        }
        try:
            self._convert = switch[vf]
            self.__vectorformat = vf
        except KeyError:
            raise VectorFormatError("vector format must be one of '1d', '2d', 'list'. given: %s" % vf)

    vectorformat = property(getVectorFormat, setVectorFormat, None, "vectorformat can be '1d', '2d' or 'list'")

    def _convertList(self, vector):
        """Converts the incoming vector to a python list."""
        return ravel(vector).tolist()

    def _convertArray1d(self, vector):
        """Converts the incoming vector to a 1d vector with shape (x,) where x
        is the number of elements."""
        return ravel(vector)

    def _convertArray2d(self, vector, column=False):
        """Converts the incoming `vector` to a 2d vector with shape (1,x), or
        (x,1) if `column` is set, where x is the number of elements."""
        a = asarray(vector)
        sh = a.shape
        # also reshape scalar values to 2d-index
        if len(sh) == 0:
            sh = (1,)
        if len(sh) == 1:
            # use reshape to add extra dimension
            if column:
                return a.reshape((sh[0], 1))
            else:
                return a.reshape((1, sh[0]))
        else:
            # vector is not 1d, return a without change
            return a

    def addField(self, label, dim):
        """Add a field to the dataset.

        A field consists of a string `label`  and a numpy ndarray of dimension
        `dim`."""
        self.data[label] = zeros((0, dim), float)
        self.endmarker[label] = 0

    def setField(self, label, arr):
        """Set the given array `arr` as the new array of field `label`,"""
        as_arr = asarray(arr)
        self.data[label] = as_arr
        self.endmarker[label] = as_arr.shape[0]

    def linkFields(self, linklist):
        """Link the length of several fields given by the list of strings
        `linklist`."""
        length = self[linklist[0]].shape[0]
        for l in linklist:
            if self[l].shape[0] != length:
                raise OutOfSyncError
        self.link = linklist

    def unlinkFields(self, unlinklist=None):
        """Remove fields from the link list or clears link given by the list of
        string `linklist`.

        This method has no effect if fields are not linked."""
        link = self.link
        if unlinklist is not None:
            for l in unlinklist:
                if l in self.link:
                    link.remove(l)
            self.link = link
        else:
            self.link = []

    def getDimension(self, label):
        """Return the dimension/number of columns for the field given by
        `label`."""
        try:
            dim = self.data[label].shape[1]
        except KeyError:
            raise KeyError('dataset field %s not found.' % label)
        return dim

    def __len__(self):
        """Return the length of the linked data fields. If no linked fields exist,
        return the length of the longest field."""
        return self.getLength()

    def getLength(self):
        """Return the length of the linked data fields. If no linked fields exist,
        return the length of the longest field."""
        if self.link == []:
            try:
                length = self.endmarker[max(self.endmarker)]
            except ValueError:
                return 0
            return length
        else:
            # all linked fields have equal length. return the length of the first.
            l = self.link[0]
            return self.endmarker[l]

    def _resize(self, label=None):
        if label:
            label = [label]
        elif self.link:
            label = self.link
        else:
            label = self.data

        for l in label:
            self.data[l] = self._resizeArray(self.data[l])

    def _resizeArray(self, a):
        """Increase the buffer size. It should always be one longer than the
        current sequence length and double on every growth step."""
        shape = list(a.shape)
        shape[0] = (shape[0] + 1) * 2
        return resize(a, shape)

    def _appendUnlinked(self, label, row):
        """Append `row` to the field array with the given `label`.

        Do not call this function from outside, use ,append() instead.
        Automatically casts vector to a 2d (or higher) shape."""
        if self.data[label].shape[0] <= self.endmarker[label]:
            self._resize(label)

        self.data[label][self.endmarker[label], :] = row
        self.endmarker[label] += 1

    def append(self, label, row):
        """Append `row` to the array given by `label`.

        If the field is linked with others, the function throws an
        `OutOfSyncError` because all linked fields always have to have the same
        length. If you want to add a row to all linked fields, use appendLink
        instead."""
        if label in self.link:
            raise OutOfSyncError
        self._appendUnlinked(label, row)

    def appendLinked(self, *args):
        """Add rows to all linked fields at once."""
        assert len(args) == len(self.link)
        for i, l in enumerate(self.link):
            self._appendUnlinked(l, args[i])

    def getLinked(self, index=None):
        """Access the dataset randomly or sequential.

        If called with `index`, the appropriate line consisting of all linked
        fields is returned and the internal marker is set to the next line.
        Otherwise the marked line is returned and the marker is moved to the
        next line."""
        if self.link == []:
            raise NoLinkedFieldsError('The dataset does not have any linked fields.')

        if index == None:
            # no index given, return the currently marked line and step marker one line forward
            index = self.index
            self.index += 1
        else:
            # return the indexed line and move marker to next line
            self.index = index + 1
        if index >= self.getLength():
            raise IndexError('index out of bounds of the dataset.')

        return [self._convert(self.data[l][index]) for l in self.link]

    def getField(self, label):
        """Return the entire field given by `label` as an array or list,
        depending on user settings."""
        if self.vectorformat == 'list':
            return self.data[label][:self.endmarker[label]].tolist()
        else:
            return self.data[label][:self.endmarker[label]]

    def hasField(self, label):
        """Tell whether the field given by `label` exists."""
        return self.data.has_key(label)

    def getFieldNames(self):
        """Return the names of the currently defined fields."""
        return self.data.keys()

    def convertField(self, label, newtype):
        """Convert the given field to a different data type."""
        try:
            self.setField(label, self.data[label].astype(newtype))
        except KeyError:
            raise KeyError('convertField: dataset field %s not found.' % label)

    def endOfData(self):
        """Tell if the end of the data set is reached."""
        return self.index == self.getLength()

    def reset(self):
        """Reset the marker to the first line."""
        self.index = 0

    def clear(self, unlinked=False):
        """Clear the dataset.

        If linked fields exist, only the linked fields will be deleted unless
        `unlinked` is set to True. If no fields are linked, all data will be
        deleted."""
        self.reset()
        keys = self.link
        if keys == [] or unlinked:
            # iterate over all fields instead
            keys = self.data

        for k in keys:
            shape = list(self.data[k].shape)
            # set to zero rows
            shape[0] = 0
            self.data[k] = zeros(shape)
            self.endmarker[k] = 0

    @classmethod
    def reconstruct(cls, filename):
        """Read an incomplete data set (option arraysonly) into the given one. """
        # FIXME: Obsolete! Kept here because of some old files...
        obj = cls(1, 1)
        for key, val in pickle.load(file(filename)).iteritems():
            obj.setField(key, val)
        return obj

    def save_pickle(self, flo, protocol=0, compact=False):
        """Save data set as pickle, removing empty space if desired."""
        if compact:
            # remove padding of zeros for each field
            for field in self.getFieldNames():
                temp = self[field][0:self.endmarker[field] + 1, :]
                self.setField(field, temp)
        Serializable.save_pickle(self, flo, protocol)

    def __reduce__(self):
        def creator():
            obj = self.__class__()
            obj.vectorformat = self.vectorformat
            return obj
        args = tuple()
        state = {
            'data': self.data,
            'link': self.link,
            'endmarker': self.endmarker,
        }
        return creator, args, state, iter([]), iter({})

    def copy(self):
        """Return a deep copy."""
        import copy
        return copy.deepcopy(self)

    def batches(self, label, n, permutation=None):
        """Yield batches of the size of n from the dataset.

        A single batch is an array of with dim columns and n rows. The last
        batch is possibly smaller.

        If permutation is given, batches are yielded in the corresponding
        order."""
        # First calculate how many batches we will have
        full_batches, rest = divmod(len(self), n)
        number_of_batches = full_batches if rest == 0 else full_batches + 1

        # We make one iterator for the startindexes ...
        startindexes = (i * n for i in xrange(number_of_batches))
        # ... and one for the stop indexes
        stopindexes = (((i + 1) * n) for i in xrange(number_of_batches - 1))
        # The last stop index is the last element of the list (last batch
        # might not be filled completely)
        stopindexes = chain(stopindexes, [len(self)])
        # Now combine them
        indexes = zip(startindexes, stopindexes)

        # Shuffle them according to the permutation if one is given
        if permutation is not None:
            indexes = [indexes[i] for i in permutation]

        for start, stop in indexes:
            yield self.data[label][start:stop]

    def randomBatches(self, label, n):
        """Like .batches(), but the order is random."""
        permutation = random.shuffle(range(len(self)))
        return self.batches(label, n, permutation)

    def replaceNansByMeans(self):
        """Replace all not-a-number entries in the dataset by the means of the
        corresponding column."""
        for d in self.data.itervalues():
            means = scipy.nansum(d[:self.getLength()], axis=0) / self.getLength()
            for i in xrange(self.getLength()):
                for j in xrange(d.dim):
                    if not scipy.isfinite(d[i, j]):
                        d[i, j] = means[j]
