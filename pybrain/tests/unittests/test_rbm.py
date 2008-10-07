"""

    >>> import scipy
    
    >>> from pybrain.structure.modules.rbm import buildRbm, invRbm
    >>> rbm = buildRbm(3, 2, 
    ...                weights=scipy.array((0, 1, 2, 3, 4, 5)))
    ... 
    >>> scipy.size(rbm.params)
    8
    >>> rbmi = invRbm(rbm)
    >>> rbmi.connections[rbmi['visible']][0].params
    array([ 0.,  3.,  1.,  4.,  2.,  5.])
    
"""

__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'
__version__ = '$Id$'


from pybrain.tests import runModuleTestSuite


if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))
