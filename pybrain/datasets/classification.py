__author__ = "Martin Felder, felder@in.tum.de"

from numpy import zeros, where, ravel, r_, single
from numpy.random import permutation
from pybrain.datasets import SupervisedDataSet, SequentialDataSet

class ClassificationDataSet(SupervisedDataSet):
    """ Specialized data set for classification data. Classes are to be numbered from 0 to nb_classes-1. """

    def __init__(self, inp, target=1, nb_classes=0, class_labels=None):
        """Initialize an empty dataset.

        `inp` is used to specify the dimensionality of the input. While the
        number of targets is given by implicitly by the training samples, it can
        also be set explicity by `nb_classes`. To give the classes names, supply
        an iterable of strings as `class_labels`."""
        # FIXME: hard to keep nClasses synchronized if appendLinked() etc. is used.
        SupervisedDataSet.__init__(self, inp, target)
        self.addField('class', 1)
        self.nClasses = nb_classes
        if len(self) > 0:
            # calculate class histogram, if we already have data
            self.calculateStatistics()
        self.convertField('target', int)
        if class_labels is None:
            self.class_labels = list(set(self.getField('target').flatten()))
        else:
            self.class_labels = class_labels
        # copy classes (may be changed into other representation)
        self.setField('class', self.getField('target'))


    @classmethod
    def load_matlab(cls, fname):
        """Create a dataset by reading a Matlab file containing one variable
        called 'data' which is an array of nSamples * nFeatures + 1 and
        contains the class in the first column."""
        from mlabwrap import mlab #@UnresolvedImport
        d = mlab.load(fname)
        return cls(d.data[:, 0], d.data[:, 1:])

    @classmethod
    def load_libsvm(cls, f):
        """Create a dataset by reading a sparse LIBSVM/SVMlight format file
        (with labels only)."""
        nFeat = 0
        # find max. number of features
        for line in f:
            n = int(line.split()[-1].split(':')[0])
            if n > nFeat:
                nFeat = n
        f.seek(0)
        labels = []
        features = []
        # read all data
        for line in f:
            # format is:
            # <class>  <featnr>:<featval>  <featnr>:<featval> ...
            # (whereby featnr starts at 1)
            if not line: break
            line = line.split()
            label = int(line[0])
            feat = []
            nextidx = 1
            for r in line[1:]:
                # construct list of features, taking care of sparsity
                (idx, val) = r.split(':')
                idx = int(idx)
                for _ in range(nextidx, idx):
                    feat.append(0.0)
                feat.append(float(val))
                nextidx = idx + 1
            for _ in range(nextidx, nFeat + 1):
                feat.append(0.0)
            features.append(feat[:])    # [:] causes copy
            labels.append([label])

        DS = cls(features, labels)
        return DS

    def __add__(self, other):
        """Adds the patterns of two datasets, if dimensions and type match."""
        if type(self) != type(other):
            raise TypeError('DataSets to be added must agree in type')
        elif self.indim != other.indim:
            raise TypeError('DataSets to be added must agree in input dimensions')
        elif self.outdim != 1 or other.outdim != 1:
            raise TypeError('Cannot add DataSets in 1-of-k representation')
        elif self.nClasses != other.nClasses:
            raise IndexError('Number of classes does not agree')
        else:
            result = self.copy()
            for pat in other:
                result.addSample(*pat)
            result.assignClasses()
        return result

    def assignClasses(self):
        """Ensure that the class field is properly defined and nClasses is set.
        """
        if len(self['class']) < len(self['target']):
            if self.outdim > 1:
                raise IndexError('Classes and 1-of-k representation out of sync!')
            else:
                self.setField('class', self.getField('target').astype(int))

        if self.nClasses <= 0:
            flat_labels = list(ravel(self['class']))
            classes = list(set(flat_labels))
            self.nClasses = len(classes)

    def calculateStatistics(self):
        """Return a class histogram."""
        self.assignClasses()
        self.classHist = {}
        flat_labels = list(ravel(self['class']))
        for class_ in range(self.nClasses):
            self.classHist[class_] = flat_labels.count(class_)
        return self.classHist

    def getClass(self, idx):
        """Return the label of given class."""
        try:
            return self.class_labels[idx]
        except IndexError:
            print("error: classes not defined yet!")

    def _convertToOneOfMany(self, bounds=(0, 1)):
        """Converts the target classes to a 1-of-k representation, retaining the
        old targets as a field `class`.

        To supply specific bounds, set the `bounds` parameter, which consists of
        target values for non-membership and membership."""
        if self.outdim != 1:
            # we already have the correct representation (hopefully...)
            return
        if self.nClasses <= 0:
            self.calculateStatistics()
        oldtarg = self.getField('target')
        newtarg = zeros([len(self), self.nClasses], dtype='Int32') + bounds[0]
        for i in range(len(self)):
            newtarg[i, int(oldtarg[i])] = bounds[1]
        self.setField('target', newtarg)
        self.setField('class', oldtarg)
        # probably better not to link field, otherwise there may be confusion
        # if getLinked() is called?
        ##self.linkFields(self.link.append('class'))

    def _convertToClassNb(self):
        """The reverse of _convertToOneOfMany. Target field is overwritten."""
        newtarg = self.getField('class')
        self.setField('target', newtarg)

    def __reduce__(self):
        _, _, state, _lst, _dct = super(ClassificationDataSet, self).__reduce__()
        creator = self.__class__
        args = self.indim, self.outdim, self.nClasses, self.class_labels
        return creator, args, state, iter([]), iter({})

    def splitByClass(self, cls_select):
        """Produce two new datasets, the first one comprising only the class
        selected (0..nClasses-1), the second one containing the remaining
        samples."""
        leftIndices, dummy = where(self['class'] == cls_select)
        rightIndices, dummy = where(self['class'] != cls_select)
        leftDs = self.copy()
        leftDs.clear()
        rightDs = leftDs.copy()
        # check which fields to split
        splitThis = []
        for f in ['input', 'target', 'class', 'importance', 'aux']:
            if self.hasField(f):
                splitThis.append(f)
        # need to synchronize input, target, and class fields
        for field in splitThis:
            leftDs.setField(field, self[field][leftIndices, :])
            leftDs.endmarker[field] = len(leftIndices)
            rightDs.setField(field, self[field][rightIndices, :])
            rightDs.endmarker[field] = len(rightIndices)
        leftDs.assignClasses()
        rightDs.assignClasses()
        return leftDs, rightDs

    def castToRegression(self, values):
        """Converts data set into a SupervisedDataSet for regression. Classes
        are used as indices into the value array given."""
        regDs = SupervisedDataSet(self.indim, 1)
        fields = self.getFieldNames()
        fields.remove('target')
        for f in fields:
            regDs.setField(f, self[f])
        regDs.setField('target', values[self['class'].astype(int)])
        return regDs


class SequenceClassificationDataSet(SequentialDataSet, ClassificationDataSet):
    """Defines a dataset for sequence classification. Each sample in the
    sequence still needs its own target value."""

    def __init__(self, inp, target, nb_classes=0, class_labels=None):
        """Initialize an empty dataset.

        `inp` is used to specify the dimensionality of the input. While the
        number of targets is given by implicitly by the training samples, it can
        also be set explicity by `nb_classes`. To give the classes names, supply
        an iterable of strings as `class_labels`."""
        # FIXME: hard to keep nClasses synchronized if appendLinked() etc. is used.
        SequentialDataSet.__init__(self, inp, target)
        # we want integer class numbers as targets
        self.convertField('target', int)
        if len(self) > 0:
            # calculate class histogram, if we already have data
            self.calculateStatistics()
        self.nClasses = nb_classes
        self.class_labels = range(self.nClasses) if class_labels is None else class_labels
        # copy classes (targets may be changed into other representation)
        self.setField('class', self.getField('target'))

    def __add__(self, other):
        """ NOT IMPLEMENTED """
        raise NotImplementedError

    def stratifiedSplit(self, testfrac=0.15, evalfrac=0):
        """Stratified random split of a sequence data set, i.e. (almost) same
        proportion of sequences in each class for all fragments. Return
        (training, test[, eval]) data sets.

        The parameter `testfrac` specifies the fraction of total sequences in
        the test dataset, while `evalfrac` specifies the fraction of sequences
        in the validation dataset. If `evalfrac` equals 0, no validationset is
        returned.

        It is assumed that the last target for each class is the class of the
        sequence. Also mind that the data will be sorted by class in the
        resulting data sets."""
        lastidx = ravel(self['sequence_index'][1:] - 1).astype(int)
        classes = ravel(self['class'][lastidx])
        trnDs = self.copy()
        trnDs.clear()
        tstDs = trnDs.copy()
        valDs = trnDs.copy()
        for c in range(self.nClasses):
            # scramble available sequences for current class
            idx, = where(classes == c)
            nCls = len(idx)
            perm = permutation(nCls).tolist()
            nTst, nVal = (int(testfrac * nCls), int(evalfrac * nCls))
            for count, ds in zip([nTst, nVal, nCls - nTst - nVal], [tstDs, valDs, trnDs]):
                for _ in range(count):
                    feat = self.getSequence(idx[perm.pop()])[0]
                    ds.newSequence()
                    for s in feat:
                        ds.addSample(s, [c])
                ds.assignClasses()
            assert perm == []
        if len(valDs) > 0:
            return trnDs, tstDs, valDs
        else:
            return trnDs, tstDs

    def getSequenceClass(self, index=None):
        """Return a flat array (or single scalar) comprising one class per
        sequence as given by last pattern in each sequence."""
        lastSeq = self.getNumSequences() - 1
        if index is None:
            classidx = r_[self['sequence_index'].astype(int)[1:, 0] - 1, len(self) - 1]
            return self['class'][classidx, 0]
        else:
            if index < lastSeq:
                return self['class'][self['sequence_index'].astype(int)[index + 1, 0] - 1, 0]
            elif index == lastSeq:
                return self['class'][len(self) - 1, 0]
            raise IndexError("Sequence index out of range!")

    def removeSequence(self, index):
        """Remove sequence (including class field) from the dataset."""
        self.assignClasses()
        self.linkFields(['input', 'target', 'class'])
        SequentialDataSet.removeSequence(self, index)
        self.unlinkFields(['class'])

    def save_netcdf(self, flo, **kwargs):
        """Save the current dataset to the given file as a netCDF dataset to be
        used with Alex Graves nnl_ndim program in
        task="sequence classification" mode."""
        # make sure classes are defined properly
        assert len(self['class']) == len(self['target'])
        if self.nClasses > 10:
            raise
        from pycdf import CDF, NC

        # need to regenerate the file name
        filename = flo.name
        flo.close()

        # Create the file. Raise the automode flag, so that
        # we do not need to worry about setting the define/data mode.
        d = CDF(filename, NC.WRITE | NC.CREATE | NC.TRUNC)
        d.automode()

        # Create 2 global attributes, one holding a string,
        # and the other one 2 floats.
        d.title = 'Sequential data exported from PyBrain (www.pybrain.org)'

        # create the dimensions
        dimsize = { 'numTimesteps':        len(self),
                    'inputPattSize':       self.indim,
                    'numLabels':           self.nClasses,
                    'numSeqs':             self.getNumSequences(),
                    'maxLabelLength':      2 }
        dims = {}
        for name, sz in dimsize.iteritems():
            dims[name] = d.def_dim(name, sz)

        # Create a netCDF record variables
        inputs = d.def_var('inputs', NC.FLOAT, (dims['numTimesteps'], dims['inputPattSize']))
        targetStrings = d.def_var('targetStrings', NC.CHAR, (dims['numSeqs'], dims['maxLabelLength']))
        seqLengths = d.def_var('seqLengths', NC.INT, (dims['numSeqs']))
        labels = d.def_var('labels', NC.CHAR, (dims['numLabels'], dims['maxLabelLength']))

        # Switch to data mode (automatic)

        # assign float and integer arrays directly
        inputs.put(self['input'].astype(single))
        # strings must be written as scalars (sucks!)
        for i in range(dimsize['numSeqs']):
            targetStrings.put_1(i, str(self.getSequenceClass(i)))
        for i in range(self.nClasses):
            labels.put_1(i, str(i))
        # need colon syntax for assigning list
        seqLengths[:] = [self.getSequenceLength(i) for i in range(self.getNumSequences())]

        # Close file
        print("wrote netCDF file " + filename)
        d.close()


if __name__ == "__main__":
    dataset = ClassificationDataSet(2, 1, class_labels=['Urd', 'Verdandi', 'Skuld'])
    dataset.appendLinked([ 0.1, 0.5 ]   , [0])
    dataset.appendLinked([ 1.2, 1.2 ]   , [1])
    dataset.appendLinked([ 1.4, 1.6 ]   , [1])
    dataset.appendLinked([ 1.6, 1.8 ]   , [1])
    dataset.appendLinked([ 0.10, 0.80 ] , [2])
    dataset.appendLinked([ 0.20, 0.90 ] , [2])

    dataset.calculateStatistics()
    print("class histogram:", dataset.classHist)
    print("# of classes:", dataset.nClasses)
    print("class 1 is: ", dataset.getClass(1))
    print("targets: ", dataset.getField('target'))
    dataset._convertToOneOfMany(bounds=[0, 1])
    print("converted targets: ")
    print(dataset.getField('target'))
    dataset._convertToClassNb()
    print("reconverted to original:", dataset.getField('target'))



