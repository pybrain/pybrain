"""
    >>> from pybrain.utilities import memoize
    >>> call_count = 0
    >>> @memoize
    ... def longComp():
    ...   global call_count
    ...   call_count += 1
    ...   return 'result'
    >>> longComp()
    'result'
    >>> call_count
    1
    >>> longComp()
    'result'
    >>> call_count
    1


Tests for Serializable
======================

    >>> from cStringIO import StringIO
    >>> s = StringIO()
    >>> p = P()
    >>> p.x = 2
    >>> p.saveToFileLike(s)
    >>> s.seek(0)
    >>> q = P.loadFromFileLike(s)
    >>> q.x
    2



Tests for permute
=================

    >>> from pybrain.utilities import permute
    >>> permute(array((0, 1, 2)), [2, 1, 0])
    array([2, 1, 0])
    >>> permute(array(((0, 0, 0), (1, 1, 1), (2, 2, 2))), (2, 0, 1))
    array([[2, 2, 2],
           [0, 0, 0],
           [1, 1, 1]])


Tests for permuteToBlocks
=========================

    >>> from pybrain.utilities import permuteToBlocks
    >>> arr = array([[0, 1, 2, 3], [4, 5 ,6 ,7], [8, 9, 10, 11], [12, 13,14, 15]])
    >>> permuteToBlocks(arr, (2, 2))
    array([  0.,   1.,   4.,   5.,   2.,   3.,   6.,   7.,   8.,   9.,  12.,
            13.,  10.,  11.,  14.,  15.])
    >>> arr = array(range(32)).reshape(2, 4, 4)
    >>> permuteToBlocks(arr, (2, 2, 2)).astype('int8').tolist()
    [0, 1, 4, 5, 16, 17, 20, 21, 2, 3, 6, 7, 18, 19, 22, 23, 8, 9, 12, 13, 24, 25, 28, 29, 10, 11, 14, 15, 26, 27, 30, 31]


"""


from scipy import array #@UnusedImport
from pybrain.utilities import Serializable
from pybrain.tests import runModuleTestSuite


class P(Serializable):

    def __getstate__(self):
        return {'x': self.x}

    def __setstate__(self, dct):
        self.x = dct['x']


if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))
