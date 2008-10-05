"""
    
    >>> from pybrain.utilities import threaded
    >>> import threading
    
    >>> result = None
    >>> def callback(c):
    ...   global result
    ...   result = c
    >>> @threaded(callback)
    ... def threadname():
    ...   return threading.currentThread().getName()
    >>> threadname()
    >>> result != threading.currentThread().getName()
    True
    
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
    
    
Tests for permuteToBlocks
=========================

    >>> from scipy import array
    >>> from pybrain.utilities import permuteToBlocks
    >>> arr = array([[0, 1, 2, 3], [4, 5 ,6 ,7], [8, 9, 10, 11], [12, 13,14, 15]])
    >>> permuteToBlocks(arr, (2, 2))
    array([0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15])

    
"""


from pybrain.utilities import Serializable
from pybrain.tests import runModuleTestSuite


class P(Serializable): 
    
    def __getstate__(self): 
        return {'x': self.x}

    def __setstate__(self, dct): 
        self.x = dct['x']


if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))
