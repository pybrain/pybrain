__author__ = "Martin Felder, felder@in.tum.de"
# $Id$

from numpy import zeros, where
from numpy.random import randint
from pybrain.datasets import SupervisedDataSet, SequentialDataSet

class ClassificationDataSet(SupervisedDataSet):
    """ Specialized data set for classification data. Classes are to be numbered from 0 to nb_classes-1. """
    
    def __init__(self, inp, target, nb_classes=0, class_labels=None):
        # FIXME: hard to keep nClasses synchronized if appendLinked() etc. is used.
        SupervisedDataSet.__init__(self, inp, target)
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
        self.setField('class', self.getField('target') )


    @classmethod
    def load_matlab(cls, fname):
        """ read Matlab file containing one variable called 'data' which is an array
        nSamples x nFeatures+1 and contains the class in the first column """
        from mlabwrap import mlab
        d=mlab.load(fname)
        return cls(d.data[:,0], d.data[:,1:])
    
    @classmethod
    def load_libsvm(cls, f):
        """ read sparse LIBSVM/SVMlight format from given (with labels only) """
        nFeat = 0
        # find max. number of features 
        for line in f:
            n = int(line.split()[-1].split(':')[0])            
            if n>nFeat:
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
                (idx,val) = r.split(':')
                idx = int(idx)
                for k in range(nextidx,idx):
                    feat.append(0.0)
                feat.append(float(val))
                nextidx = idx+1
            for k in range(nextidx,nFeat+1):
                feat.append(0.0)
            features.append(feat[:])    # [:] causes copy
            labels.append([label])
        
        DS = cls(features, labels)
        return DS
 
    def calculateStatistics(self):
        """ calculate class histogram """
        flat_labels = list( self.getField('target').flatten() )
        classes       = list(set( flat_labels ))
        self.nClasses = len(classes)
        self.classHist = {}
        for class_ in classes:
            self.classHist[class_] = flat_labels.count(class_)
        return self.classHist

    def getClass(self,idx):
        try:
            return self.class_labels[idx]
        except IndexError:
            print "error: classes not defined yet!" 

    def _convertToOneOfMany(self, bounds=[0,1]):
        """ converts the target classes to a 1-of-k representation, retaining the old targets as a field 'class' """
        if self.outdim != 1:
            # we already have the correct representation (hopefully...)
            return
        if self.nClasses <=0:
            self.calculateStatistics()
        oldtarg = self.getField('target')
        newtarg = zeros([len(self), self.nClasses],dtype='Int32') + bounds[0]
        for i in range(len(self)):
            newtarg[i,int(oldtarg[i])] = bounds[1]
        self.setField('target',newtarg)
        self.setField('class', oldtarg)
        # probably better not to link field, otherwise there may be confusion 
        # if getLinked() is called?
        ##self.linkFields(self.link.append('class'))

    def _convertToClassNb(self):
        """ the reverse of _convertToOneOfMany. 1-of-k field is overwritten.  """
        newtarg = self.getField('class')
        self.setField('target', newtarg)
        
    def _equalizeClasses(self):
        """ Re-sample dataset to ensure equal number of data for each class """
        n = max(self.classHist)
        if self.outdim == 1:
            classes = self['target']
        else:
            classes = self['class']
        for cls, nVal in self.classHist.iteritems():
            idx = where(classes == cls)
            i = randint(len(idx), size=n-nVal)
            i = idx[i]
            
    def __reduce__(self):
        _, _, state, lst, dct = super(ClassificationDataSet, self).__reduce__()
        creator = self.__class__
        args = self.indim, self.outdim, self.nClasses, self.class_labels
        return creator, args, state, [], {}
            
    def splitByClass(self, cls_select):
        """ produce two new datasets, the first one comprising only the class selected (0..nClasses-1),
        the second one containing the remaining samples """
        leftIndices, dummy = where(self['class'] == cls_select)
        rightIndices, dummy = where(self['class'] != cls_select)        
        leftDs = self.copy()
        rightDs = self.copy()
        leftDs.clear()
        rightDs.clear()
        index = 0
        # need to synchronize input, target, and class fields
        for field in ['input','target','class']:
            leftDs.setField(field, self[field][leftIndices,:])
            leftDs.endmarker[field] = len(leftIndices)
            rightDs.setField(field, self[field][rightIndices,:])
            rightDs.endmarker[field] = len(rightIndices)
        
        return leftDs, rightDs
    
    def castToRegression(self,values):
        """ Converts data set into a SupervisedDataSet, for regression. Classes are changed into
        the value array given."""
        regDs = SupervisedDataSet(self.indim, 1)
        regDs.setField('input', self['input'])
        regDs.setField('target', values[self['class'].astype(int)])
        return regDs
    
 
class SequenceClassificationDataSet(SequentialDataSet, ClassificationDataSet):
    
    def __init__(self, inp, target, nb_classes=0, class_labels=None):
        # FIXME: hard to keep nClasses synchronized if appendLinked() etc. is used.
        SequentialDataSet.__init__(self, inp, target)
        self.nClasses = nb_classes
        if len(self) > 0:
            # calculate class histogram, if we already have data
            self.calculateStatistics()
        self.convertField('target',int)
        if class_labels is None:
            self.class_labels = list(set(self.getField('target').flatten()))
        else:
            self.class_labels = class_labels
        # copy classes (may be changed into other representation)
        self.setField('class', self.getField('target') )


if __name__ == "__main__":
    dataset = ClassificationDataSet(2,1, class_labels=['Urd','Verdandi','Skuld'])
    dataset.appendLinked( [ 0.1, 0.5 ]   , [0] )
    dataset.appendLinked( [ 1.2, 1.2 ]   , [1] )
    dataset.appendLinked( [ 1.4, 1.6 ]   , [1] )
    dataset.appendLinked( [ 1.6, 1.8 ]   , [1] )
    dataset.appendLinked( [ 0.10, 0.80 ] , [2] )
    dataset.appendLinked( [ 0.20, 0.90 ] , [2] )

    dataset.calculateStatistics()
    print "class histogram:", dataset.classHist
    print "# of classes:", dataset.nClasses
    print "class 1 is: ", dataset.getClass(1)
    print "targets: ", dataset.getField('target')
    dataset._convertToOneOfMany(bounds=[0,1])
    print "converted targets: "
    print dataset.getField('target')
    dataset._convertToClassNb()
    print "reconverted to original:", dataset.getField('target')

    
