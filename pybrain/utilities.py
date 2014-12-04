from __future__ import print_function

__author__ = 'Tom Schaul, tom@idsia.ch; Justin Bayer, bayerj@in.tum.de'

import gc
import pickle
import logging
import threading
import os
import operator

from itertools import count
from math import sqrt
from random import random, choice

from scipy import where, array, exp, zeros, size, mat, median
from functools import reduce

# file extension for load/save protocol mapping
known_extensions = {
    'mat': 'matlab',
    'txt': 'ascii',
    'svm': 'libsvm',
    'pkl': 'pickle',
    'nc' : 'netcdf' }


def abstractMethod():
    """ This should be called when an abstract method is called that should have been
    implemented by a subclass. It should not be called in situations where no implementation
    (i.e. a 'pass' behavior) is acceptable. """
    raise NotImplementedError('Method not implemented!')


def drawIndex(probs, tolerant=False):
    """ Draws an index given an array of probabilities.

    :key tolerant: if set to True, the array is normalized to sum to 1.  """
    if not sum(probs) < 1.00001 or not sum(probs) > 0.99999:
        if tolerant:
            probs /= sum(probs)
        else:
            print((probs, 1 - sum(probs)))
            raise ValueError()
    r = random()
    s = 0
    for i, p in enumerate(probs):
        s += p
        if s > r:
            return i
    return choice(list(range(len(probs))))


def drawGibbs(vals, temperature=1.):
    """ Return the index of the sample drawn by a softmax (Gibbs). """
    if temperature == 0:
        # randomly pick one of the values with the max value.
        m = max(vals)
        best = []
        for i, v in enumerate(vals):
            if v == m:
                best.append(i)
        return choice(best)
    else:
        temp = vals / temperature

        # make sure we keep the exponential bounded (between +20 and -20)
        temp += 20 - max(temp)
        if min(temp) < -20:
            for i, v in enumerate(temp):
                if v < -20:
                    temp[i] = -20
        temp = exp(temp)
        temp /= sum(temp)
        return drawIndex(temp)


def iterCombinations(tup):
    """ all possible of integer tuples of the same dimension than tup, and each component being
    positive and strictly inferior to the corresponding entry in tup. """
    if len(tup) == 1:
        for i in range(tup[0]):
            yield (i,)
    elif len(tup) > 1:
        for prefix in iterCombinations(tup[:-1]):
            for i in range(tup[-1]):
                yield tuple(list(prefix) + [i])


def setAllArgs(obj, argdict):
    """ set all those internal variables which have the same name than an entry in the
    given object's dictionary.
    This function can be useful for quick initializations. """

    xmlstore = isinstance(obj, XMLBuildable)
    for n in list(argdict.keys()):
        if hasattr(obj, n):
            setattr(obj, n, argdict[n])
            if xmlstore:
                obj.argdict[n] = argdict[n]
        else:
            print(('Warning: parameter name', n, 'not found!'))
            if xmlstore:
                if not hasattr(obj, '_unknown_argdict'):
                    obj._unknown_argdict = {}
                obj._unknown_argdict[n] = argdict[n]


def linscale(d, lim):
    """ utility function to linearly scale array d to the interval defined by lim """
    return (d - d.min())*(lim[1] - lim[0]) + lim[0]


def percentError(out, true):
    """ return percentage of mismatch between out and target values (lists and arrays accepted) """
    arrout = array(out).flatten()
    wrong = where(arrout != array(true).flatten())[0].size
    return 100. * float(wrong) / float(arrout.size)


def formatFromExtension(fname):
    """Tries to infer a protocol from the file extension."""
    _base, ext = os.path.splitext(fname)
    if not ext:
        return None
    try:
        format = known_extensions[ext.replace('.', '')]
    except KeyError:
        format = None
    return format


class XMLBuildable(object):
    """ subclasses of this can be losslessly stored in XML, and
    automatically reconstructed on reading. For this they need to store
    their construction arguments in the variable <argdict>. """

    argdict = None

    def setArgs(self, **argdict):
        if not self.argdict:
            self.argdict = {}
        setAllArgs(self, argdict)


class Serializable(object):
    """Class that implements shortcuts to serialize an object.

    Serialization is done by various formats. At the moment, only 'pickle' is
    supported.
    """

    def saveToFileLike(self, flo, format=None, **kwargs):
        """Save the object to a given file like object in the given format.
        """
        format = 'pickle' if format is None else format
        save = getattr(self, "save_%s" % format, None)
        if save is None:
            raise ValueError("Unknown format '%s'." % format)
        save(flo, **kwargs)

    @classmethod
    def loadFromFileLike(cls, flo, format=None):
        """Load the object to a given file like object with the given protocol.
        """
        format = 'pickle' if format is None else format
        load = getattr(cls, "load_%s" % format, None)
        if load is None:
            raise ValueError("Unknown format '%s'." % format)
        return load(flo)

    def saveToFile(self, filename, format=None, **kwargs):
        """Save the object to file given by filename."""
        if format is None:
            # try to derive protocol from file extension
            format = formatFromExtension(filename)
        with file(filename, 'wb') as fp:
            self.saveToFileLike(fp, format, **kwargs)

    @classmethod
    def loadFromFile(cls, filename, format=None):
        """Return an instance of the class that is saved in the file with the
        given filename in the specified format."""
        if format is None:
            # try to derive protocol from file extension
            format = formatFromExtension(filename)
        with file(filename, 'rbU') as fp:
            obj = cls.loadFromFileLike(fp, format)
            obj.filename = filename
            return obj

    def save_pickle(self, flo, protocol=0):
        pickle.dump(self, flo, protocol)

    @classmethod
    def load_pickle(cls, flo):
        return pickle.load(flo)


class Named(XMLBuildable):
    """Class whose objects are guaranteed to have a unique name."""

    _nameIds = count(0)

    def getName(self):
        logging.warning("Deprecated, use .name property instead.")
        return self.name

    def setName(self, newname):
        logging.warning("Deprecated, use .name property instead.")
        self.name = newname

    def _getName(self):
        """Returns the name, which is generated if it has not been already."""
        if self._name is None:
            self._name = self._generateName()
        return self._name

    def _setName(self, newname):
        """Change name to newname. Uniqueness is not guaranteed anymore."""
        self._name = newname

    _name = None
    name = property(_getName, _setName)

    def _generateName(self):
        """Return a unique name for this object."""
        return "%s-%i" % (self.__class__.__name__, next(self._nameIds))

    def __repr__(self):
        """ The default representation of a named object is its name. """
        return "<%s '%s'>" % (self.__class__.__name__, self.name)


def fListToString(a_list, a_precision=3):
    """ Returns a string representing a list of floats with a given precision """
    from numpy import around
    s_list = ", ".join(("%g" % around(x, a_precision)).ljust(a_precision+3)
                      for x in a_list)
    return "[%s]" % s_list

def tupleRemoveItem(tup, index):
    """ remove the item at position index of the tuple and return a new tuple. """
    l = list(tup)
    return tuple(l[:index] + l[index + 1:])


def confidenceIntervalSize(stdev, nbsamples):
    """ Determine the size of the confidence interval, given the standard deviation and the number of samples.
    t-test-percentile: 97.5%, infinitely many degrees of freedom,
    therefore on the two-sided interval: 95% """
    # CHECKME: for better precision, maybe get the percentile dynamically, from the scipy library?
    return 2 * 1.98 * stdev / sqrt(nbsamples)


def trace(func):
    def inner(*args, **kwargs):
        print(("%s: %s, %s" % (func.__name__, args, kwargs)))
        return func(*args, **kwargs)
    return inner


def threaded(callback=lambda * args, **kwargs: None, daemonic=False):
    """Decorate  a function to run in its own thread and report the result
    by calling callback with it."""
    def innerDecorator(func):
        def inner(*args, **kwargs):
            target = lambda: callback(func(*args, **kwargs))
            t = threading.Thread(target=target)
            t.setDaemon(daemonic)
            t.start()
        return inner
    return innerDecorator


def garbagecollect(func):
    """Decorate a function to invoke the garbage collector after each execution.
    """
    def inner(*args, **kwargs):
        result = func(*args, **kwargs)
        gc.collect()
        return result
    return inner


def memoize(func):
    """Decorate a function to 'memoize' results by holding it in a cache that
    maps call arguments to returns."""
    cache = {}
    def inner(*args, **kwargs):
        # Dictionaries and lists are unhashable
        args = tuple(args)
        # Make a set for checking in the cache, since the order of
        # .iteritems() is undefined
        kwargs_set = frozenset(iter(kwargs.items()))
        if (args, kwargs_set) in cache:
            result = cache[args, kwargs_set]
        else:
            result = func(*args, **kwargs)
            cache[args, kwargs_set] = result
        return result
    return inner


def storeCallResults(obj, verbose=False):
    """Pseudo-decorate an object to store all evaluations of the function in the returned list."""
    results = []
    oldcall = obj.__class__.__call__
    def newcall(*args, **kwargs):
        result = oldcall(*args, **kwargs)
        results.append(result)
        if verbose:
            print(result)
        return result
    obj.__class__.__call__ = newcall
    return results


def multiEvaluate(repeat):
    """Decorate a function to evaluate repeatedly with the same arguments, and return the average result """
    def decorator(func):
        def inner(*args, **kwargs):
            result = 0.
            for dummy in range(repeat):
                result += func(*args, **kwargs)
            return result / repeat
        return inner
    return decorator


def _import(name):
    """Return module from a package.

    These two are equivalent:

        > from package import module as bar
        > bar = _import('package.module')

    """
    mod = __import__(name)
    components = name.split('.')
    for comp in components[1:]:
        try:
            mod = getattr(mod, comp)
        except AttributeError:
            raise ImportError("No module named %s" % mod)
    return mod


# tools for binary Gray code manipulation:

def int2gray(i):
    """ Returns the value of an integer in Gray encoding."""
    return i ^ (i >> 1)


def gray2int(g, size):
    """ Transforms a Gray code back into an integer. """
    res = 0
    for i in reversed(list(range(size))):
        gi = (g >> i) % 2
        if i == size - 1:
            bi = gi
        else:
            bi = bi ^ gi
        res += bi * 2 ** i
    return res


def asBinary(i):
    """ Produces a string from an integer's binary representation.
    (preceding zeros removed). """
    if i > 1:
        if i % 2 == 1:
            return asBinary(i >> 1) + '1'
        else:
            return asBinary(i >> 1) + '0'
    else:
        return str(i)


def one_to_n(val, maxval):
    """ Returns a 1-in-n binary encoding of a non-negative integer. """
    a = zeros(maxval, float)
    a[val] = 1.
    return a


def n_to_one(arr):
    """ Returns the reverse of a 1-in-n binary encoding. """
    return where(arr == 1)[0][0]


def canonicClassString(x):
    """ the __class__ attribute changed from old-style to new-style classes... """
    if isinstance(x, object):
        return repr(x.__class__).split("'")[1]
    else:
        return repr(x.__class__)


def decrementAny(tup):
    """ the closest tuples to tup: decrementing by 1 along any dimension.
    Never go into negatives though. """
    res = []
    for i, x in enumerate(tup):
        if x > 0:
            res.append(tuple(list(tup[:i]) + [x - 1] + list(tup[i + 1:])))
    return res


def reachable(stepFunction, start, destinations, _alreadyseen=None):
    """ Determines the subset of destinations that can be reached from a set of starting positions,
    while using stepFunction (which produces a list of neighbor states) to navigate.
    Uses breadth-first search.
    Returns a dictionary with reachable destinations and their distances.
    """
    if len(start) == 0 or len(destinations) == 0:
        return {}
    if _alreadyseen is None:
        _alreadyseen = []
    _alreadyseen.extend(start)

    # dict with distances to destinations
    res = {}
    for s in start:
        if s in destinations:
            res[s] = 0
            start.remove(s)

    # do one step
    new = set()
    for s in start:
        new.update(stepFunction(s))
    new.difference_update(_alreadyseen)
    ndestinations = list(destinations)

    for s in list(new):
        if s in destinations:
            res[s] = 1
            new.remove(s)
            ndestinations.remove(s)
            _alreadyseen.append(s)

    # recursively do the rest
    deeper = reachable(stepFunction, new, ndestinations, _alreadyseen)

    # adjust distances
    for k, val in list(deeper.items()):
        res[k] = val + 1
    return res


def flood(stepFunction, fullSet, initSet, relevant=None):
    """ Returns a list of elements of fullSet linked to some element of initSet
    through the neighborhood-setFunction (which must be defined on all elements of fullSet).

    :key relevant: (optional) list of relevant elements: stop once all relevant elements are found.
    """
    if fullSet is None:
        flooded = set(initSet)
    else:
        full = set(fullSet)
        flooded = full.intersection(set(initSet))
        if relevant is None:
            relevant = full.copy()
    if relevant:
        relevant = set(relevant)

    change = flooded.copy()
    while len(change)>0:
        new = set()
        for m in change:
            if fullSet is None:
                new.update(stepFunction(m))
            else:
                new.update(full.intersection(stepFunction(m)))
        change = new.difference(flooded)
        flooded.update(change)
        if relevant is not None and relevant.issubset(flooded):
            break
    return list(flooded)


def crossproduct(ss, row=None, level=0):
    """Returns the cross-product of the sets given in `ss`."""
    if row is None:
        row = []
    if len(ss) > 1:
        return reduce(operator.add,
                      [crossproduct(ss[1:], row + [i], level + 1) for i in ss[0]])
    else:
        return [row + [i] for i in ss[0]]


def permute(arr, permutation):
    """Return an array like arr but with elements permuted.

    Only the first dimension is permuted, which makes it possible to permute
    blocks of the input.

    arr can be anything as long as it's indexable."""
    return array([arr[i] for i in permutation])


def permuteToBlocks(arr, blockshape):
    """Permute an array so that it consists of linearized blocks.

    Example: A two-dimensional array of the form

        0  1  2  3
        4  5  6  7
        8  9  10 11
        12 13 14 15

    would be turned into an array like this with (2, 2) blocks:

        0 1 4 5 2 3 6 7 8 9 12 13 10 11 14 15
    """
    if len(blockshape) < 2:
        raise ValueError("Need more than one dimension.")
    elif len(blockshape) == 2:
        blockheight, blockwidth = blockshape
        return permuteToBlocks2d(arr, blockheight, blockwidth)
    elif len(blockshape) == 3:
        blockdepth, blockheight, blockwidth = blockshape
        return permuteToBlocks3d(arr, blockdepth, blockheight, blockwidth)
    else:
        raise NotImplementedError("Only for dimensions 2 and 3.")


def permuteToBlocks3d(arr, blockdepth, blockheight, blockwidth):
    depth, height, width = arr.shape
    arr_ = arr.reshape(height * depth, width)
    arr_ = permuteToBlocks2d(arr_, blockheight, blockwidth)
    arr_.shape = depth, height * width
    return permuteToBlocks2d(arr_, blockdepth, blockwidth * blockheight)


def permuteToBlocks2d(arr, blockheight, blockwidth):
    _height, width = arr.shape
    arr = arr.flatten()
    new = zeros(size(arr))
    for i in range(size(arr)):
        blockx = (i % width) / blockwidth
        blocky = i / width / blockheight
        blockoffset = blocky * width / blockwidth + blockx
        blockoffset *= blockwidth * blockheight
        inblockx = i % blockwidth
        inblocky = (i / width) % blockheight
        j = blockoffset + inblocky * blockwidth + inblockx
        new[j] = arr[i]
    return new


def triu2flat(m):
    """ Flattens an upper triangular matrix, returning a vector of the
    non-zero elements. """
    dim = m.shape[0]
    res = zeros(dim * (dim + 1) / 2)
    index = 0
    for row in range(dim):
        res[index:index + dim - row] = m[row, row:]
        index += dim - row
    return res


def flat2triu(a, dim):
    """ Produces an upper triangular matrix of dimension dim from the elements of the given vector. """
    res = zeros((dim, dim))
    index = 0
    for row in range(dim):
        res[row, row:] = a[index:index + dim - row]
        index += dim - row
    return res


def blockList2Matrix(l):
    """ Converts a list of matrices into a corresponding big block-diagonal one. """
    dims = [m.shape[0] for m in l]
    s = sum(dims)
    res = zeros((s, s))
    index = 0
    for i in range(len(l)):
        d = dims[i]
        m = l[i]
        res[index:index + d, index:index + d] = m
        index += d
    return res


def blockCombine(l):
    """ Produce a matrix from a list of lists of its components. """
    l = [list(map(mat, row)) for row in l]
    hdims = [m.shape[1] for m in l[0]]
    hs = sum(hdims)
    vdims = [row[0].shape[0] for row in l]
    vs = sum(vdims)
    res = zeros((hs, vs))
    vindex = 0
    for i, row in enumerate(l):
        hindex = 0
        for j, m in enumerate(row):
            res[vindex:vindex + vdims[i], hindex:hindex + hdims[j]] = m
            hindex += hdims[j]
        vindex += vdims[i]
    return res


def avgFoundAfter(decreasingTargetValues, listsOfActualValues, batchSize=1, useMedian=False):
    """ Determine the average number of steps to reach a certain value (for the first time),
    given a list of value sequences.
    If a value is not always encountered, the length of the longest sequence is used.
    Returns an array. """
    from scipy import sum
    numLists = len(listsOfActualValues)
    longest = max(list(map(len, listsOfActualValues)))
    # gather a list of indices of first encounters
    res = [[0] for _ in range(numLists)]
    for tval in decreasingTargetValues:
        for li, l in enumerate(listsOfActualValues):
            lres = res[li]
            found = False
            for i in range(lres[-1], len(l)):
                if l[i] <= tval:
                    lres.append(i)
                    found = True
                    break
            if not found:
                lres.append(longest)
    tmp = array(res)
    if useMedian:
        resx = median(tmp, axis=0)[1:]
    else:
        resx = sum(tmp, axis=0)[1:] / float(numLists)
    return resx * batchSize


class DivergenceError(Exception):
    """ Raised when an algorithm diverges. """


def matchingDict(d, selection, require_existence=False):
    """ Determines if the dictionary d conforms to the specified selection,
    i.e. if a (key, x) is in the selection, then if key is in d as well it must be x
    or contained in x (if x is a list). """
    for k, v in list(selection.items()):
        if k in d:
            if isinstance(v, list):
                if d[k] not in v:
                    return False
            else:
                if d[k] != v:
                    return False
        elif require_existence:
            return False
    return True


def subDict(d, allowedkeys, flip=False):
    """ Returns a new dictionary with a subset of the entries of d
    that have on of the (dis-)allowed keys."""
    res = {}
    for k, v in list(d.items()):
        if (k in allowedkeys) ^ flip:
            res[k] = v
    return res


def dictCombinations(listdict):
    """ Iterates over dictionaries that go through every possible combination
    of key-value pairs as specified in the lists of values for each key in listdict."""
    listdict = listdict.copy()
    if len(listdict) == 0:
        return [{}]
    k, vs = listdict.popitem()
    res = dictCombinations(listdict)
    if isinstance(vs, list) or isinstance(vs, tuple):
        res = [dict(d, **{k:v}) for d in res for v in sorted(set(vs))]
    else:
        res = [dict(d, **{k:vs}) for d in res]
    return res


def r_argmax(v):
    """ Acts like scipy argmax, but break ties randomly. """
    if len(v) == 1:
        return 0
    maxbid = max(v)
    maxbidders = [i for (i, b) in enumerate(v) if b==maxbid]
    return choice(maxbidders)

def all_argmax(x):
    """ Return the indices of all values that are equal to the maximum: no breaking ties. """
    m = max(x)
    return [i for i, v in enumerate(x) if v == m]

def dense_orth(dim):
    """ Constructs a dense orthogonal matrix. """
    from scipy import rand
    from scipy.linalg import orth
    return orth(rand(dim, dim))
    
def sparse_orth(d):
    """ Constructs a sparse orthogonal matrix.
    
    The method is described in:
    Gi-Sang Cheon et al., Constructions for the sparsest orthogonal matrices,
    Bull. Korean Math. Soc 36 (1999) No.1 pp.199-129
    """
    from scipy.sparse import eye
    from scipy import r_, pi, sin, cos
    if d%2 == 0:
        seq = r_[0:d:2,1:d-1:2]
    else:
        seq = r_[0:d-1:2,1:d:2]
    Q = eye(d,d).tocsc()
    for i in seq:
        theta = random() * 2 * pi
        flip = (random() - 0.5)>0;
        Qi = eye(d,d).tocsc()
        Qi[i,i] = cos(theta)
        Qi[(i+1),i] = sin(theta)
        if flip > 0:
            Qi[i,(i+1)] = -sin(theta)
            Qi[(i+1),(i+1)] = cos(theta)
        else:
            Qi[i,(i+1)] = sin(theta)
            Qi[(i+1),(i+1)] = -cos(theta)            
        Q = Q*Qi;
    return Q

def xhash(arr):
    """ Hashing function for arrays. Use with care. """
    import hashlib
    return hashlib.sha1(arr).hexdigest()

def binArr2int(arr):
    """ Convert a binary array into its (long) integer representation. """
    from numpy import packbits
    tmp2 = packbits(arr.astype(int))
    return sum(val * 256 ** i for i, val in enumerate(tmp2[::-1])) 
        
def uniqueArrays(vs):
    """ create a set of arrays """
    resdic = {}
    for v in vs:
        resdic[xhash(v)] = v
    return list(resdic.values())    
    

def seedit(seed=0):
    """ Fixed seed makes for repeatability, but there may be two different
    random number generators involved. """
    import random
    import numpy
    random.seed(seed)
    numpy.random.seed(seed)


    
def weightedUtest(g1, w1, g2, w2):
    """ Determines the confidence level of the assertion:
    'The values of g2 are higher than those of g1'.  
    (adapted from the scipy.stats version)
    
    Twist: here the elements of each group have associated weights, 
    corresponding to how often they are present (i.e. two identical entries with 
    weight w are equivalent to one entry with weight 2w).
    Reference: "Studies in Continuous Black-box Optimization", Schaul, 2011 [appendix B].
    
    TODO: make more efficient for large sets. 
    """
    from scipy.stats.distributions import norm
    import numpy
    n1 = sum(w1)
    n2 = sum(w2)
    u1 = 0.
    for x1, wx1 in zip(g1, w1):
        for x2, wx2 in zip(g2, w2):
            if x1 == x2:
                u1 += 0.5 * wx1 * wx2
            elif x1 > x2:
                u1 += wx1 * wx2
    mu = n1*n2/2.
    sigu = numpy.sqrt(n1*n2*(n1+n2+1)/12.)
    z = (u1 - mu) / sigu
    conf = norm.cdf(z)
    return conf 

