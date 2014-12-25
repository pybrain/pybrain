# -*- coding: utf-8 -*-

"""
    >>> from scipy import array
    >>> from pybrain import datasets
    >>> from copy import deepcopy
    >>> d = datasets.dataset.DataSet()
    >>> d.addField('input', 2)
    >>> type(d.data['input'])
	<type 'numpy.ndarray'>
	
    >>> len(d.data['input'])
    0
	
    >>> x, y = d.data['input'].shape
	>>> str(x)
	0
	>>> str(y)
    2
	
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
    >>> print('\\n'.join(repr(b) for b in d.batches('input', 2, p)))
    array([[ 6.,  6.],
           [ 7.,  7.]])
    array([[ 4.,  4.],
           [ 5.,  5.]])
    array([[ 2.,  2.],
           [ 3.,  3.]])
    array([[ 0.,  0.],
           [ 1.,  1.]])


Serialization
=============

    >>> from cStringIO import StringIO


UnsupervisedDataSet
-----------------

    >>> d = datasets.UnsupervisedDataSet(2)
    >>> d.addSample([0,0])
    >>> d.addSample([0,1])
    >>> d.addSample([1,0])
    >>> d.addSample([1,1])
    >>> for sample in d:
    ...   print(sample)
    ...
    [array([ 0.,  0.])]
    [array([ 0.,  1.])]
    [array([ 1.,  0.])]
    [array([ 1.,  1.])]






ClassificationDataSet
---------------------

    >>> class_labels = 'Urd', 'Verdandi', 'Skuld'
    >>> d = datasets.ClassificationDataSet(2,1, class_labels=class_labels)
    >>> d.appendLinked( [ 0.1, 0.5 ]   , [0] )
    >>> d.appendLinked( [ 1.2, 1.2 ]   , [1] )
    >>> d.appendLinked( [ 1.4, 1.6 ]   , [1] )
    >>> d.appendLinked( [ 1.6, 1.8 ]   , [1] )
    >>> d.appendLinked( [ 0.10, 0.80 ] , [2] )
    >>> d.appendLinked( [ 0.20, 0.90 ] , [2] )

    >>> saveInvariant(d)
    True


ImportanceDataSet
-----------------


SequentialDataSet
-----------------

      >>> d = datasets.SequentialDataSet(0, 1)
      >>> d.addSample([],[0])
      >>> d.addSample([],[1])
      >>> d.addSample([],[0])
      >>> d.addSample([],[1])
      >>> d.addSample([],[0])
      >>> d.addSample([],[1])
      >>> d.newSequence()
      >>> d.addSample([],[0])
      >>> d.addSample([],[1])
      >>> d.addSample([],[0])
      >>> d.addSample([],[1])
      >>> d.addSample([],[0])
      >>> d.addSample([],[1])

      >>> saveInvariant(d)
      True


ReinforcementDataSet
--------------------

    >>> d = datasets.ReinforcementDataSet(1, 1)
    >>> d.addSample([1,], [1,], [1,])
    >>> d.addSample([1,], [1,], [1,])
    >>> d.addSample([1,], [1,], [1,])
    >>> saveInvariant(d)
    True



"""


__author__ = 'Justin Bayer, bayerj@in.tum.de'


from io import StringIO

from pybrain.tests import runModuleTestSuite


def saveInvariant(dataset):
    # Save and reconstruct
    s = StringIO()
    dataset.saveToFileLike(s)
    s.seek(0)
    reconstructed = dataset.__class__.loadFromFileLike(s)

    orig_array_data = sorted(dataset.data.items())
    rec_array_data = sorted(reconstructed.data.items())
    equal = True
    for (k, v), (k_, v_) in zip(orig_array_data, rec_array_data):
        if k != k_:
            print(("Differing keys: %s <=> %s" % (list(dataset.dataset.keys()),
                                                 list(rec_array_data.dataset.keys()))))
            equal = False
            break
        if not (v == v_).all():
            print(("Differing values for %s" % k))
            print(v)
            print(v_)
            equal = False
            break

    if not equal:
        return False

    rec_dict = reconstructed.__dict__
    orig_dict = dataset.__dict__

    del rec_dict['_convert']
    del orig_dict['_convert']
    del rec_dict['data']
    del orig_dict['data']

    if rec_dict == orig_dict:
        return True
    else:
        print(rec_dict)
        print(orig_dict)
        return False


if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))
