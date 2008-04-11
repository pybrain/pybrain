# $Id$
__author__ = 'Michael Isik'

from   scipy import zeros,empty
from   numpy import dot,sqrt,exp,array,where,apply_along_axis
import numpy
import copy

from module import Module

class KT:
    """ Class providing some constants for kernel-types """
    LINEAR  = 0
    POLY    = 1
    RBF     = 2
    SIGMOID = 3





class Dumpable(object):
    """ Base Class that provides saving and loading of objects of derived Classes
        to/from a file. """
    @classmethod
    def _importCPickle(cls):
        try: cPickle
        except NameError:
            import cPickle
            global cPickle

    def dumpToFile( self, filename ):
        """ Save the serialized data of the current object to a file.
            @param filename : Name of file to save to
        """
        Dumpable._importCPickle()
        fh = open(filename,"w")
        cPickle.dump(self,fh)
        fh.close()

    @classmethod
    def loadFromFile( cls, filename ):
        """ Construct an object from the serialized data of a file, and set self
            to this object.
            @param filename : Name of file to load from
        """
        Dumpable._importCPickle()
        fh = open(filename,"r")
        obj = cPickle.load(fh)
        fh.close()
        return obj


def vectorizeX(func):
    """ Decorator for functions that expect one input vector x,
        but should also be able to expect an array of vectors.

        If x is a single vector, the regular function will be called and its
        return value be forwarded.

        If x is an array of vectors, func will be called for each of these vectors.
        An array will be returned that holds the return values of each call to func.
    """
    def inner_func(self,x):
        x = array(x)
        if   x.ndim == 1: return func(self,x)
        elif x.ndim == 2:
            def self_func(x): return func(self,x)
            res = apply_along_axis(self_func,1,x)
#            res = []
#            for xi in x:
#                res.append( func(self,xi) )
            return res
        else: raise Exception("Invalid number of dimensions")

    return inner_func


class SVM(Module,Dumpable):
    """ The simple main module class for support vector machines. It is meant
        to separate the input vectors of TWO classes. For data with more than
        two classes use the MCSVM module.

        Train the SVM module by associating it to an instance of SVMTrainer.

        You can save an instance of SVM to a file by calling the inhertited
        method saveToFile(). For loading use loadFromFile().
    """

    def __init__( self, indim, outdim, kernel = KT.RBF, **kwargs ):
        """ Initialize the SVM module.

            @param  indim: Dimensionality of input data
            @param outdim: Dimensionality of output data. Should always be set to 1.
            @param kernel: May either be a kernel type constant supplied by the
                           KT class ( e.g. KT.POLY ) or an instance of a subclass
                           of the kernel class. This instance will just function
                           as prototype, hence, a copy will be made.
            @param **kwargs: These kwargs will be redirected to the kernel class.
                             See Kernel for more info.
        """
        Module.__init__( self, indim, outdim )

        self._w      = None
        self._alpha  = None
        self._beta   = 0.
        if isinstance(kernel, Kernel):
            self._kernel = copy.deepcopy(kernel)
        else:
            self._kernel = {
                KT.LINEAR  : LinearKernel  (),
                KT.POLY    : PolyKernel    (),
                KT.RBF     : RBFKernel     (),
                KT.SIGMOID : SigmoidKernel ()
            }[kernel]
        self.setParams(**kwargs)

    def setParams(self,**kwargs):
        """ Sets the module's parameters. These will mostly be redirected
            to the kernel. See Kernel for more information about kwargs.
        """
        for key, value in kwargs.items():
            if key in ("verbose", "ver", "v"):
                self._verbosity = value
            else:
                self._kernel.setParams(**{key:value})

    def _forwardImplementation( self, inbuf, outbuf ):
        """ The forward pass method of a trained support vector machine.
            It classifies the input vector stored in inbuf, and writes the
            class label to outbuf. Note that the length of the outbuf vector
            should always be 1.

            @param  inbuf: An input vector
            @param outbuf: An output vector of length self.outdim. The output
                           will be written to this vector.
        """
        outbuf[0] = self.classify(inbuf)

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        """ dummy backward implementation """
        inerr[:] = outerr


    def _setData( self, dataset ):
        """ Set the training data needed by the kernel by supplying an instance
            of SupervisedDataSet
        """
        l = dataset.getLength()
        X = dataset.getField("input")
        Y = dataset.getField("target").flatten()
        self._setDataXY(X,Y)


    def _setDataXY(self, X, Y):
        """ Same as _setData except you have to supply two vectors of input and
            target data instead of a SupervisedDataSet.

            @param X : Vector of input vectors
            @param Y : Vector of target class labels
        """
        classes       = list(set(Y))
        self._classes = classes
        if len(classes) > 2: raise Exception("Too many classes detected. Use MCSVM instead.")
        where0, = where( Y == classes[0] )
        where1, = where( Y == classes[1] )
        mapped_Y = empty(len(Y),float)
        mapped_Y[where0] = -1
        mapped_Y[where1] =  1
        self._kernel.setData(X,mapped_Y)
        self._alpha = zeros(self._kernel.l)


    @vectorizeX
    def rawOutput( self, x ):
        """ Returns the raw output of the trained SVM model, without classifying it.
            The output is the scaled distance of the input vector x to the separating
            hyperplane.

            This distance is calculated by:

                f(x) = sum_over_l(  alpha[i] * Y[i] * k( X[i], x )  ) - beta

            where:
                - x     : the input vector
                - l     : number of training samples stored inside the kernel
                - X     : array of input vectors of the training set
                - Y     : array of target values of the training set (-1 or 1)
                - alpha : array of multiplier that were trained
                - beta  : trained threshold
        """
        y = 0
        alpha  = self._alpha
        k      = self._kernel.k
        X      = self._kernel._X
        Y      = self._kernel._Y

#        for i in range(self._kernel.l):
#            y += alpha[i] * Y[i] * k(X[i],x)

        y = sum( alpha * Y * k(X,x) )

        y -= self._beta

        return y

    def getClass(self,idx):
        return self._classes[idx]


    @vectorizeX
    def classify( self, x ):
        """ Returns the classified output of the trained SVM model regarding the
            input vector x.
            This is done by calculating the raw output of the svm for input x,
            and then mapping the sign of this value to the respective class label.

            rawOutput <  0  -->  self._classes[0] will be returned
            rawOutput >= 0  -->  self._classes[1] will be returned
        """
        return self._classes[ self.rawOutput(x) >= 0 ]

    def rawOutputToClass( self, rawout ):
        return self._classes[ rawout >= 0 ]

    def classToRawOutput( self, class_ ):
        try:
            idx = self._classes.index(class_)
            if   idx == 0: return -1.
            elif idx == 1: return  1.
        except ValueError:
            return None



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
#            elif key in ("one_against_one"):
#                self._one_against_one = value
#            elif key in ("one_against_all"):
#                self._one_against_one = not value

        self._sub_kernel_kwargs = kwargs

#        if self._one_against_one:
#            self._setData      = self._setDataOneAgainstOne
#            self.classify      = self._classifyOneAgainstOne
#            self.getSubModules = self._getSubModulesOneAgainstOne
#        else:
#            self._setData      = self._setDataOneAgainstAll
#            self.classify      = self._classifyOneAgainstAll
#            self.getSubModules = self._getSubModulesOneAgainstAll

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


class Kernel:
    """ Base class for Kernels. A derived class must at least implement the
        kernel method named k.

        The kernel stores all the training samples, because they will be needed for
        calculating the raw outputs of new input vectors, which is
        fundamental for classification.
        The number of samples will be stored in the public attribute l.
        Input vectors will be stored in the _X array, while target values
        reside in _Y. Note that each target value is elements of { -1 , 1 }

        For better performance a subclass should also overwrite the generic
        _calcQRow() method with an adapted version for the specific kernel function.
        See the RBFKernel._calcQRow()
    """

    def __init__(self,**kwargs):
        """ Initializes the kernel.

            @param **kwargs: see setParams()
        """
        self._X = []
        self._Y = []
        self._degree = 3.
        self._gamma  = 1.
        self._coef0  = 0.
        self.setParams(**kwargs)

        self._cache = {}

    def setParams(self, **kwargs):
        """ Set kernels parameters.

            @param **kwargs: allowed keys:
                * degree : Degree parameter used by some kernels. ( default = 3 )
                * coef0  : Comparable to a bias of some kernels.  ( default = 0 )
                * gamma  : Scaling factor used by some kernels.   ( default = 1 )
        """
        for key, value in kwargs.items():
            if   key in ("degree") : self._degree = float( value )
            elif key in ("gamma" ) : self._gamma  = float( value )
            elif key in ("coef0")  : self._coef0  = float( value )




    def k(self,x1,x2):
        """ Abstract kernel method. Must be implemented.

            Calculates the inner product of the input vectors x1 and x2,
            after transforming them into feature space:
                < phi(x1), phi(x2) >

            @param x1: A single sample or an array of samples
            @param x2: A single sample
        """
        raise NotImplementedError()

    def kIdx(self,i,j):
        """ Kernel function, that wrapps the native k method. Instead of
            specifying the input vectors itself, indices are supplied.
            These indices adress the corresponding training vectors of the
            training set stored by the kernel
        """
        X = self._X
        return self.k( X[i], X[j] )




    def getQRow(self, i):
        """ Returns the i'th row of the Q matrix. Uses caching for optimization.
            The Q matrix is defined by:
                Q[i][j] = Y[i] * Y[j] * k( X[i] , X[j] )

            This function uses caching, because the calculation of one row is
            very expensive. The actual calculation of the row happens inside
            _calcQRow()

            todo: So far no memory limitation for the cache is implemented.
                  Implement a lru cache, which has a maximum amount of stored
                  rows.
        """
        # todo: implement shrinking support

        cache = self._cache
        if not cache.has_key(i):
            row = self._calcQRow(i)
            cache[i] = row
            return row
        else:
            return cache[i]


    def _calcQRow(self,i):
        """ Actually calculates the i'th row of the Q matrix. Is called by the
            getQRow() method. See getQRow() documentation for a description of Q.

            Should be overwritten with an more efficient method of calculating 
            the row of the particular kernel.
        """
        # overwrite this function with an optimized one
        X    = self._X
        k    = self.k
        xi   = X[i]
        s    = self._Y * self._Y[i]
        q_i  = array( [ k(xi,xj) for xj in X ], dtype=float ) * s
        return q_i




    def setData(self,X,Y):
        """ Sets the kernel data.
            @param X : 2D numpy array with inputvectors of same length
            @param Y : 1D numpy array with corresponding target values of the inputvectors
        """
        self._X = X
        self._Y = Y
        self.l  = len(Y)

        # calculate diagonal
        self._QD = array( [ self.kIdx(i,i) for i in range(self.l) ], dtype=float )


    def getXDim(self):
        """ Returns dimensionality of input space. """
        return len( self._X[0] )

    def isExplicit(self):
        """ Returns, wheter this kernel object has an explicit phi method,
            for transforming an input vector into feature space.

            Always returns False. ExplicitKernel overrides method.
        """
        return False

class LinearKernel(Kernel):
    """ The linear kernel class """

    def k( self, x1, x2 ):
        """ Kernel method that just calculates the inner product of x1 and x2:
                k(x1,x2) = <x1,x2>
        """
        return dot( x1, x2 )


class PolyKernel(Kernel):
    """ The polynomial kernel class """

    def k( self, x1, x2 ):
        """ Kernel method that calculates:
               k(x1,x2) = ( gamma *  <x1,x2> + coef0 ) ^ degree
        """
        return ( self._gamma * dot(x1,x2) + self._coef0 ) ** self._degree



class RBFKernel(Kernel):
    """ The radial basis function Kernel class """
    def k( self, x1, x2 ):
        """ Kernel method that calculates:
                k(x1,x2) = exp( - gamma * |x1-x2| ^ 2 )
        """
        xd = x1-x2
        if xd.ndim <= 1: return exp( - self._gamma * dot(xd,xd) )
        else:            return exp( - self._gamma * numpy.sum( xd**2, axis=1 ) )

    def _calcQRow(self,i):
        """ Efficient way of calculating the i'th row of the Q matrix """
        s   = self._Y * self._Y[i]
        xd  = self._X - self._X[i]
#        k_i = exp( numpy.negative( numpy.sum( xd**2, axis=1 ) ) / ( 2. * self._sigma**2 ) )
        k_i = exp( numpy.negative( self._gamma * numpy.sum( xd**2, axis=1 ) ) )
        q_i = k_i * s
        return q_i



class SigmoidKernel(Kernel):
    def k(self,x1,x2):
        """ Kernel method that calculates:
                k(x1,x2) = tanh( gamma * <x1,x2> + coef0 )
        """
        return numpy.tanh(self._gamma * dot(x1,x2) + self._coef0 )

    def _calcQRow(self,i):
        """ Efficient way of calculating the i'th row of the Q matrix """
        k_i = numpy.tanh( self._gamma * dot( self._X, self._X[i] ) + self._coef0 )
        s   = self._Y * self._Y[i]
        q_i = k_i * s
        return q_i




class ExplicitKernel(Kernel):
    """ Base class for Kernels with explicit feature function phi. These can be
        used for analyzing the kernel\'s feature space.
    """

    def k(self,x1,x2):
        """ Kernel function, that calculates the resulting value by
                1. transforming the input vectors into feature space using the
                   feature function phi()
                2. calculating the inner product of these two feature vectors

                        k(x1,x2) = < phi(x1), phi(x2) >
        """
        return dot( self.phi(x1), self.phi(x2) )

    def phi(self,x):
        """ Abstract feature function phi. Transforms an input x vector into
            feature space.

            Must be implemented.
        """
        raise NotImplementedError()

    def isExplicit(self):
        """ Always returns True, because all explicit kernels possess a feature
            function phi.
        """
        return True

    def getFeatureSpaceDim(self):
        """ Returns dimensionality of feature space defined by the phi-function. """
        return len( self.phi( self._X[0] ) )


class SimplePolyKernel(ExplicitKernel):
    """ A simple polynomial kernel of degree 2, coef0 0 and gamma 1 with
        explicit feature function. It can just be used for 2 dimensional input
        data. Use it for demonstrations of its 3D feature space. """
    def phi(self,x):
        """ Feature function:
                phi( x1 ) = ( x[0] ^ 2 , sqrt(2) * x[0] * x[1] , x[1] ^ 2 )
        """
        return array( [ x[0]**2, sqrt(2)*x[0]*x[1], x[1]**2 ] )

    def k(self,x1,x2):
        """ Efficient way of calculating k. """
        return dot(x1,x2) ** 2


