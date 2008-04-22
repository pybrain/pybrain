from __future__ import with_statement

__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'
# $Id$

import types
import cPickle

from itertools import chain

from scipy import zeros, resize, ravel, asarray

from pybrain.utilities import abstractMethod

class OutOfSyncError(Exception): pass
class VectorFormatError(Exception): pass
class NoLinkedFieldsError(Exception): pass

class DataSet(object):
    """ DataSet is a general base class for other data set classes (e.g. SupervisedDataSet, SequentialDataSet, ...).
        It consists of several fields. A field is a NumPy array with a label (a string) attached to it. Fields can
        be linked together, which means they must have the same length. """
        
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
        """ string representation of a dataset. """
        s = ""
        for key in self.data:
            s = s + key + ": dim" + str(self.data[key].shape) + "\n" + str(self.data[key][:self.endmarker[key]]) + "\n\n" 
        return s
            
    def __getitem__(self, field):
        """ returns the given field. """
        return self.getField(field)
        
    def __iter__(self):
        """ makes the DataSet iteratable. For instance, use: "for sample in dataset: ... """
        self.reset()
        while not self.endOfData():
            yield self.getLinked()

    def getVectorFormat(self):
        """ returns the current vector format. use the property vectorformat. """
        return self.__vectorformat
        
    def setVectorFormat(self, vf):
        """ determine which format to use for returning vectors. use the property vectorformat.
            @param type: possible types are '1d', '2d', 'list' 
                  '1d' - example: array([1,2,3])
                  '2d' - example: array([[1,2,3]])
                'list' - example: [1,2,3]
                'none' - no conversion
         """
        switch = {
            '1d': self._convertArray1d,
            '2d': self._convertArray2d,
            'list': self._convertList,
            'none': lambda(x):x
        }
        try:
            self._convert = switch[vf]
            self.__vectorformat = vf
        except KeyError:
            raise VectorFormatError("vector format must be one of '1d', '2d', 'list'. given: %s" % vf)
    
    vectorformat = property(getVectorFormat, setVectorFormat, None, "vectorformat can be '1d', '2d' or 'list'")

    def _convertList(self, vector):
        """ converts the incoming vector to a python list. """
        return ravel(vector).tolist()
        
    def _convertArray1d(self, vector):
        """ converts the incoming vector to a 1d vector with shape (x,) where x is the number of elements. """
        return ravel(vector)
        
    def _convertArray2d(self, vector, column=False):
        """ converts the incoming vector to a 2d vector with shape (1,x), or (x,1) if column is set, where
            x is the number of elements.
            @param vector: the object to reshape (can be array, scalar, list) 
            @param column: if set to True, the result is a column rather than a row vector. """
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
        """ adds a field to the dataset. a field consists of a label (string) and an 
            numpy ndarray.
            @param label: name of the field (string)
            @param dim: the column dimension of the array. """
        self.data[label] = zeros((0, dim), float)
        self.endmarker[label] = 0
        
    def setField(self, label, arr):
        """ sets the given array as the new array of field 'label'
            @param label: the name of the field
            @param arr: the new array for that field """
        as_arr = asarray(arr)
        self.data[label] = as_arr
        self.endmarker[label] = as_arr.shape[0]
                
    def linkFields(self, linklist):
        """ links the length of several fields. These fields can be manipulated
            together more easily, and they must always have the same length.
            @param linklist: a list of field labels that should be linked together """
        length = self[linklist[0]].shape[0]
        for l in linklist:
            if self[l].shape[0] != length:
                raise OutOfSyncError
        self.link = linklist
        
    def unlinkFields(self, unlinklist=None):
        """ removes fields from the link list, or clears link. No effect if fields are not linked.
            @param linklist: a list of field labels that should be linked together """
        link = self.link
        if unlinklist is not None:
            for l in unlinklist:
                if l in self.link:
                    link.remove(l)
            self.link = link
        else:
            self.link = []
        
    def getDimension(self, label):
        """ returns the dimension (= number of columns) for the given field.
            @param label: the label for which the dimension is returned """
        try:
            dim = self.data[label].shape[1]
        except KeyError:
            raise KeyError('dataset field %s not found.' % label)
        return dim
        
    def __len__(self):
        """ returns the length of the linked data fields. if no linked fields exist, 
            returns the length of the longest field. """
        return self.getLength()
        
    def getLength(self):
        """ see __len__ """
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
        """ increase the buffer size. It should always be one longer than the
            current sequence length and double on every growth step.
        """
        shape = list(a.shape)
        shape[0] = (shape[0]+1) * 2
        return resize(a, shape)
            
    def _appendUnlinked(self, label, row):
        """ internal function, which appends a row to the field array with the
            given label. Do not call this function from outside, use append
            instead. Automatically casts vector to a 2d (or higher) shape. 
            @param label: appends the row to the field with that name
            @param row: the row (automatically converted to 2d array) to append """
        if self.data[label].shape[0] <= self.endmarker[label]:
            self._resize(label)
         
        self.data[label][self.endmarker[label],:] = row
        self.endmarker[label] += 1

    def append(self, label, row):
        """ appends a row to the array of the given label. If the field is linked with others,
            the function throws the OutOfSyncError exception, because all linked fields always have
            to have the same length. If you want to add a row to all linked fields, use appendLink 
            instead. 
            @param label: appends the row to the field with that name 
            @param row: the row (automatically converted to 2d array) to append """
        if label in self.link:
            raise OutOfSyncError
        self._appendUnlinked(label, row)
            
    def appendLinked(self, *args):
        """ This function is used to add a row to all linked fields at once. It has a variable
            argument list, which takes exactly as many arguments as there are linked fields. The
            rows are added in the order of the self.link list. 
            @param args: expects number of arguments equal to number of linked fields """
        
        assert len(args) == len(self.link)
        for i,l in enumerate(self.link):
            self._appendUnlinked(l, args[i])
     
    def getLinked(self, index=None):
        """ This function allows both random and sequential access to the dataset. If called 
            with an index, the appropriate line consisting of all linked fields is
            returned and the internal marker is set to the next line. If called without an index
            (index=None), the marked line is returned and the marker is moved to the next line. 
            @param index: the index of the row to be returned. if index=None, the current row is returned """
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
            
        return (map(self._convert, [self.data[l][index] for l in self.link]))    

    def getField(self, label):
        """ Return the entire field as an array or list, depending on user settings.
            @param label: the name of the field that should be returned """
        if self.vectorformat == 'list':
            return self.data[label][:self.endmarker[label]].tolist()
        else:
            return self.data[label][:self.endmarker[label]]
    
    def hasField(self, label):
        """ Checks whether specified field exists.
            @param label: the name of the field  """
        return self.data.has_key(label)
        
    def getFieldNames(self):
        """ Returns names of the currently defined fields """
        return self.data.keys()
    
    def convertField(self, label, newtype):
        """ Converts the given field to a different data type """
        try:
            self.setField(label, self.data[label].astype(newtype))
        except KeyError:
            raise KeyError('convertField: dataset field %s not found.' % label)
            
    def endOfData(self):
        """ returns True if the end of the data set is reached (use with iteration). """
        return self.index == self.getLength()

    def reset(self):
        """ resets the marker to the first line. """
        self.index = 0
    
    def clear(self, unlinked=False):
        """ clears the dataset. if linked fields exist, only the linked fields will be
            deleted, unless unlinked is set to True. if no fields are linked, all data
            will be deleted. """
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

    def saveToFile(self, filename, **kwargs):
        """Save the current dataset to the given filename."""
        fp = file(filename, 'w+')
        self._saveToFileLike(fp, **kwargs)
        fp.close()
            
    def _saveToFileLike(self, flo, protocol=0, arraysonly=False ):
        """Save the current dataset into the given file like object."""
        if arraysonly:
            # failsave version
            cPickle.dump(self.data, flo, protocol=protocol)
        else:
            package = {
                'initial': self._initialValues(),
                'dict': self._dumpDict(),
            }
            cPickle.dump(package, flo, protocol=protocol)

    @classmethod
    def loadFromFile(cls, filename):
        fp = file(filename)
        tmp = cls._loadFromFileLike(fp)
        fp.close()
        return tmp

    @classmethod
    def _loadFromFileLike(cls, flo):
        package = cPickle.load(flo)
        args, kwargs = package['initial']
        obj = cls(*args, **kwargs)
        obj.__dict__.update(package['dict'])
        return obj
        
    def _initialValues(self):
        abstractMethod()
        
    def _dumpDict(self):
        return dict((k, v) for k, v in self.__dict__.items() 
                    if type(v) is types.FunctionType)
        
    def copy(self):
        """ deep copy. """
        import copy
        return copy.deepcopy(self)
        
    def batches(self, label, n, permutation=None):
        """Yield batches of the size of n from the dataset. 

        A single batch is an array of with dim columns and n rows. The last 
        batch is possibly smaller.
        
        If permutation is given, batches are yielded in the corresponding 
        order.
        """
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
        permutation = shuffle(range(len(self)))
        return self.batches(label, n, permutation)
                      
        