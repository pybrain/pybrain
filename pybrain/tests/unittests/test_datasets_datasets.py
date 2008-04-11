#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""

    >>> from scipy import array
    >>> from pybrain.datasets.dataset import DataSet
    >>> d = DataSet()
    >>> d.addField('input', 2)
    >>> d.data['input']
    array([], shape=(0, 2), dtype=...)
    
Build up a DataSet for testing:
    
    >>> d.append('input', (array((0, 0))))
    >>> d.append('input', (array((1, 1))))
    >>> d.append('input', (array((2, 2))))
    >>> d.append('input', (array((3, 3))))
    >>> d.append('input', (array((4, 4))))
    >>> d.append('input', (array((5, 5))))
    >>> d.append('input', (array((6, 6))))
    >>> d.append('input', (array((7, 7))))
    
    >>> list(d.batches('input', 3))
    [array([[ 0.,  0.],
               [ 1.,  1.],
               [ 2.,  2.]]), array([[ 3.,  3.],
               [ 4.,  4.],
               [ 5.,  5.]]), array([[ 6.,  6.],
               [ 7.,  7.]])]
                          
    >>> list(d.batches('input', 2))
    [array([[ 0.,  0.],
               [ 1.,  1.]]), array([[ 2.,  2.],
               [ 3.,  3.]]), array([[ 4.,  4.],
               [ 5.,  5.]]), array([[ 6.,  6.],
               [ 7.,  7.]])]
                          
    >>> p = reversed(range(4))
    >>> print '\\n'.join(repr(b) for b in d.batches('input', 2, p))
    array([[ 6.,  6.],
           [ 7.,  7.]])
    array([[ 4.,  4.],
           [ 5.,  5.]])
    array([[ 2.,  2.],
           [ 3.,  3.]])
    array([[ 0.,  0.],
           [ 1.,  1.]])
    
    
    

"""

__author__ = 'Justin Bayer, bayerj@in.tum.de'

from pybrain.tests import runModuleTestSuite

if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))
