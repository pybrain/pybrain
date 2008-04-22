# $Id$
__author__ = 'Michael Isik'


import collections
import numpy
from numpy import array, where, zeros, empty

from module import Module
from pybrain.structure.modules.svm import SVM, KT, Dumpable, vectorizeX


class AbstractMCSVM(Module,Dumpable):
    """ Module class for training data with more than two classes.
        The one-against-one approach is used, in which a set of traditional SVM
        modules is hosted, one module for each possible pair of classes.

        On classification, each sub module classifies the input vector,
        and gives a vote for one of its two class labels. The label with the most
        votes wins, and will be returned by the MCSVM.classify() method.
    """
    def __init__( self, indim, outdim, kernel = KT.RBF, **kwargs ):
        """ Initialize the Module. See the SVM.__init__() documentation for
            more information.

            @param kernel : This kernel will be used for all sub modules
        """
        Module.__init__( self, indim, outdim )
        self._one_against_one = True
        self.setParams(**kwargs)
        self._kernel = kernel


    def setParams(self, **kwargs):
        """ Set the module's parameters. See SVM.setParams() for more information.
        """
        for key, value in kwargs.items():
            if key in ("verbose", "ver", "v"):
                self._verbosity = value

        self._sub_kernel_kwargs = kwargs


    def classifyDataSet( self, dataset ):
        """ Return the predicted class for an entire data set """
        dataset.reset()
        self.reset()
        out = zeros(len(dataset))
        for i, sample in enumerate(dataset):
            # FIXME: Can we always assume that sample[0] is the input data?
            out[i] = self.classify(sample[0])
        self.reset()
        dataset.reset()
        return out

    @vectorizeX
    def rawOutputs(self,x):
        """ Returns a vector of raw outputs regarding input vector x
            of all sub modules, in the order
            [0-1, 0-2, ..., 0-n, 1-2, 1-3, ..., (n-1)-n].
            The length of this vector can be retrieved by the nSubModules()
            method.
        """
        margins = empty( len(self._sub_modules), float )
        for i,sub_module in enumerate(self._sub_modules):
            margins[i] = sub_module.rawOutput(x)
        return margins


    def _forwardImplementation( self, inbuf, outbuf ):
        """ Forward pass method, that classifies the input vector inbuf
            and writes the label to outbuf. See SVM._forwardImplementation()
            documentation.
        """
        outbuf[0] = self.classify(inbuf)

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        """ dummy backward implementation """
        inerr[:] = outerr

    def getSubModules(self):
        raise NotImplementedError

    def classify( self, x ):
        raise NotImplementedError


    def nSubModules(self):
        """ Returns the number of sub modules """
        return len(self._sub_modules)

    def getClass(self,idx):
        return self._classes[idx]

    def _setData( self, dataset ):
        raise NotImplementedError

class MCSVMOneAgainstOne(AbstractMCSVM):
    def getSubModules(self):
        return self._sub_modules

    def _setData( self, dataset ):
        """ Set the training data by supplying an instance of SupervisedDataSet.

            The samples will be separated into subsets of same class. Then
            each SVM sub module will be fed with data containing samples of
            just the two classes that were assigned to it.
        """
        self._X = dataset.getField("input")
        self._Y = dataset.getField("target").flatten()
        X       = self._X
        Y       = self._Y
        indim   = self.indim
        outdim  = self.outdim
        classes        = list( set(Y) )
        self._classes  = classes
        classnum       = len(classes)

        sub_modules = []
        self._sub_modules = sub_modules
        for i in range(classnum):
            for j in range( i+1, classnum ):
                sub_module = SVM( indim, outdim, self._kernel, **self._sub_kernel_kwargs )

                where0, = where( Y == classes[i] )
                where1, = where( Y == classes[j] )

                sub_X = numpy.append( X[where0], X[where1], axis=0 )
                sub_Y = numpy.append( Y[where0], Y[where1])
                sub_module._setDataXY(sub_X, sub_Y)
                sub_modules.append( sub_module )

    @vectorizeX
    def classify( self, x ):
        """ Classify input vector x by using the one-against-one strategy.
            See class description for more information.
        """

        votes = collections.defaultdict( lambda: 0 )

        for sub_module in self._sub_modules:
            c = sub_module.classify(x)
            votes[c] += 1

        counts = array(votes.values())
        max_idx = numpy.argmax(counts)
        winner_class = votes.keys()[max_idx]
        return winner_class

class MCSVMOneAgainstAll(AbstractMCSVM):
    def getSubModules(self):
        return self._sub_modules.values()

    def _setData( self, dataset ):
        self._X = dataset.getField("input")
        self._Y = dataset.getField("target").flatten()
        X       = self._X
        Y       = self._Y
        indim   = self.indim
        outdim  = self.outdim
        classes        = list( set(Y) )
        self._classes  = classes
        n_classes      = len(classes)

        sub_modules = {}
        self._sub_modules = sub_modules
        for i in range(n_classes):
            sub_module = SVM( indim, outdim, self._kernel, **self._sub_kernel_kwargs )

            where0, = where( Y == classes[i] )
            where1, = where( Y != classes[i] )

            sub_X = numpy.append( X[where0], X[where1], axis=0 )
            sub_Y = numpy.append( [True for j in range(len(where0))], [False for j in range(len(where1))] )
#            print sub_X, sub_Y
            sub_module._setDataXY(sub_X, sub_Y)
            sub_modules[classes[i]] = sub_module


    @vectorizeX
    def classify( self, x ):

        votes = collections.defaultdict( lambda: 0 )

        for class_,sub_module in self._sub_modules.items():
#            print sub_module.classify(x)
            if sub_module.classify(x):
                votes[class_] += 1

        if not len(votes):
            raise Exception("Classification failed")# zzzzzzzz was hier tun?
#            return None
#            raise ErrorNotImplementedError("Classification failed")

        counts  = array(votes.values())
        max_idx = numpy.argmax(counts)
        winner_class = votes.keys()[max_idx]
        return winner_class




MCSVM = MCSVMOneAgainstOne


class MCSVMrawPair(MCSVM):
    """ Multi-class SVM that returns raw outputs for all pairwise sub-classifiers
    """

    def __init__( self, indim, outdim, kernel = KT.RBF, **kernel_kwargs ):
        Module.__init__( self, indim, outdim )
        self._sub_kernel_kwargs = kernel_kwargs
        self._kernel = kernel

    def _forwardImplementation( self, inbuf, outbuf ):
        outbuf[:] = self.rawOutputs(inbuf)



