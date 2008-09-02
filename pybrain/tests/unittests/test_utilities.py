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
    

Tests for substitute
====================

Hack Environment Variables
--------------------------

Since substitution can be deactivated via environment variables, we have to 
hack them first. This is no real testing.

    >>> import os
    >>> no_substitute = os.environ.get('PYBRAIN_NO_SUBSTITUTE', False)
    >>> if no_substitute: del os.environ['PYBRAIN_NO_SUBSTITUTE']


Tests
-----

First define a function that is supposed to be substituted.
    
    >>> @substitute('pybrain.tests.auxiliary.otherfunc')
    ... def onefunc():
    ...    print "I am onefunc."

Then assert that it really happens:
    
    >>> onefunc()
    I am the other func.
    
Test that attaching builtins to classes works as expected.

    >>> class MyNumber(int):
    ...   @substitute('math.log')
    ...   def log(self): pass

    >>> import math
    >>> MyNumber(2).log()
    0.69314718055994529

Unhack Environment Variables
----------------------------
    
Recover PYBRAIN_NO_SUBSTITUTE flag

    >>> if no_substitute: os.environ['PYBRAIN_NO_SUBSTITUTE'] = no_substitute
   
   
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
