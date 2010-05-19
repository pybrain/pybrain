# -*- coding: utf-8 -*-

__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'
__version__ = '$Id$'


from pybrain.datasets.dataset import DataSet


class UnsupervisedDataSet(DataSet):
    """UnsupervisedDataSets have a single field 'sample'."""

    def __init__(self, dim):
        """Initialize an empty unsupervised dataset.

        Pass `dim` to specify the dimensionality of the samples."""
        super(UnsupervisedDataSet, self).__init__()
        self.addField('sample', dim)
        self.linkFields(['sample'])
        self.dim = dim

        # reset the index marker
        self.index = 0

    def __reduce__(self):
        _, _, state, _, _ = super(UnsupervisedDataSet, self).__reduce__()
        creator = self.__class__
        args = (self.dim,)
        return creator, args, state, iter([]), iter({})

    def addSample(self, sample):
        self.appendLinked(sample)

    def getSample(self, index):
        return self.getLinked(index)
