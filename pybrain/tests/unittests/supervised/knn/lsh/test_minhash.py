# -*- coding: utf-8 -*-

"""
Internal Tests:

    >>> from scipy import array
    >>> from pybrain.supervised.knn.lsh.minhash import arrayPermutation
    >>> permutation = array([4, 3, 2, 1, 0])
    >>> permute = arrayPermutation(permutation)
    >>> permute(array([5, 2, 3, 1, 4]))
    array([4, 1, 3, 2, 5])

    >>> from pybrain.supervised.knn.lsh.minhash import jacardCoefficient
    >>> a = array([0, 0, 0, 1])
    >>> b = array([1, 1, 1, 1])
    >>> c = array([1, 1, 0, 1])
    >>> d = array([0, 0, 1, 1])
    >>> jacardCoefficient(a, b)
    0.25
    >>> jacardCoefficient(b, c)
    0.75
    >>> jacardCoefficient(a, a)
    1.0
    >>> jacardCoefficient(a, d)
    0.75


Example Usage:

    >>> from pybrain.supervised.knn.lsh.minhash import MinHash

We need to specify the length of the inputs and how many permutations should be
used:

    >>> m = MinHash(5, 1)

The permutation is initialized randomly

    >>> m.permutations
    array([...])

But for the tests, we will justify it to our means:

    >>> m.permutations = array([[0, 1, 2, 4, 3], [0, 1, 2, 3, 4]])

So let's put in some values that will hash to the same bucket

    >>> m.put(array([1, 1, 1, 1, 1]), 'red')

Some "introspection" to check if everything went right

    Changed this test to work with python 2 AND 3.
    Old correct output: defaultdict(<function <lambda> at ...>, {(0, 0): [(array([...True], dtype=bool), 'red')]})

    >>> import pprint
    >>> pprint.pprint(dict(m.buckets))
    {(0, 0): [(array([ True,  True,  True,  True,  True], dtype=bool), 'red')]}

    >>> m._hash(array([1, 1, 0, 0, 0]))
    (0, 0)

Put another one in

    >>> m.put(array([1, 1, 1, 1, 0]), 'red')

An check if this one is favored above the other

    >>> m.knn(array([1, 1, 0, 0, 0]), 1)
    [(array([... True, False], dtype=bool), 'red')]
    >>> m.knn(array([1, 1, 0, 0, 0]), 1)
    [(array([... True, False], dtype=bool), 'red')]



Let's make a hash that returns nothing

    >>> m.knn(array([0, 0, 0, 0, 0]), 1)
    []

"""

from pybrain.tests import runModuleTestSuite

if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))
