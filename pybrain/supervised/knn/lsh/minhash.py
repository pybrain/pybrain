


"""Module that provides functionality for locality sensitive hashing in hamming
spaces."""


__author__ = 'Justin Bayer, bayer.justin@googlemail.com'
__version__ = '$Id$'


from collections import defaultdict
from scipy import array
from numpy.random import permutation


def arrayPermutation(permutation):
    """Return a permutation function.

    The function permutes any array as specified by the supplied permutation.
    """
    assert permutation.ndim == 1, \
           "Only one dimensional permutaton arrays are supported"

    def permute(arr):
        assert arr.ndim == 1, "Only one dimensional arrays are supported"
        assert arr.shape == permutation.shape, "Array shapes don't match"
        return array([arr[i] for i in permutation])

    return permute


def jacardCoefficient(a, b):
    """Return the Jacard coefficient of a and b.

    The jacard coefficient is defined as the overlap between two sets: the sum
    of all equal elements divided by the size of the sets.

    Mind that a and b must b in Hamming space, so every element must either be
    1 or 0.
    """
    if a.shape != b.shape:
        raise ValueError("Arrays must be of same shape")

    length = a.shape[0]
    a = a.astype(bool)
    b = b.astype(bool)

    return float((a == b).sum()) / length


class MinHash(object):
    """Class for probabilistic hashing of items in the hamming space.

    Introduced in

        E. Cohen. Size-Estimation Framework with Applications to
        Transitive Closure and Reachability. Journal of Computer and System
        Sciences 55 (1997): 441-453"""

    def __setPermutations(self, permutations):
        self._permutations = permutations
        self._permFuncs = [arrayPermutation(i) for i in permutations]

    def __getPermutations(self):
        return self._permutations

    permutations = property(__getPermutations,
                            __setPermutations)

    def __init__(self, dim, nPermutations):
        """Create a hash structure that can hold arrays of size dim and
        hashes with nPermutations permutations.

        The number of buckets is dim * nPermutations."""
        self.dim = dim
        self.permutations = array([permutation(dim)
                                   for _ in range(nPermutations)])

        self.buckets = defaultdict(lambda: [])

    def _firstOne(self, arr):
        """Return the index of the first 1 in the array."""
        for i, elem in enumerate(arr):
            if elem == 1:
                return i
        return i + 1

    def _checkItem(self, item):
        if item.ndim != 1:
            raise ValueError("Only one dimensional arrays are supported")
        if item.shape != (self.dim,):
            raise ValueError("Array has wrong size")

    def _hash(self, item):
        """Return a hash for item based on the internal permutations.

        That hash is a tuple of ints.
        """
        self._checkItem(item)

        result = []
        for perm in self._permFuncs:
            permuted = perm(item)
            result.append(self._firstOne(permuted))
        return tuple(result)

    def put(self, item, satellite):
        """Put an item into the hash structure and attach any object satellite
        to it."""
        self._checkItem(item)

        item = item.astype(bool)
        bucket = self._hash(item)
        self.buckets[bucket].append((item, satellite))


    def knn(self, item, k):
        """Return the k nearest neighbours of the item in the current hash.

        Mind that the probabilistic nature of the data structure might not
        return a nearest neighbor at all.
        """
        self._checkItem(item)

        candidates = self.buckets[self._hash(item)]
        candidates.sort(key=lambda x: jacardCoefficient(x[0], item),
                        reverse=True)
        return candidates[:k]


