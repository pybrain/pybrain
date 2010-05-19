# -*- coding: utf-8 -*-

""" WARNING: this file is a construction site. The classes are currently placeholders for stuff to come. """

__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'
__version__ = '$Id$'

import operator
import scipy

try:
    from arac.pybrainbridge import _FeedForwardNetwork #@UnresolvedImport
except:
    _FeedForwardNetwork = object
from pybrain.structure.modules.mdrnnlayer import MdrnnLayer
from pybrain.structure import LinearLayer
from pybrain.structure.connections.permutation import PermutationConnection
from pybrain.utilities import crossproduct, permute, permuteToBlocks


class _Mdrnn(_FeedForwardNetwork):

    def __init__(self, timedim, shape,
                 hiddendim, outsize, blockshape=None, name=None,
                 inlayerclass=LinearLayer, outlayerclass=LinearLayer):
        super(_Mdrnn, self).__init__()
        # Initialize necessary member variables
        self.timedim = timedim
        self.shape = shape
        self.hiddendim = hiddendim
        self.outsize = outsize
        self.blockshape = blockshape
        self.indim = reduce(operator.mul, shape, 1)
        self.blocksize = reduce(operator.mul, blockshape, 1)
        self.sequenceLength = self.indim / self.blocksize
        self.inlayerclass = inlayerclass
        self.outlayerclass = outlayerclass

        # Build up topology
        self._buildTopology()

    def _makeMdrnnLayer(self):
        """Return an MdrnnLayer suitable for this network."""
        return MdrnnLayer(self.timedim, self.shape, self.hiddendim,
                          self.outsize, self.blockshape)

    def _standardPermutation(self):
        """Return the permutation of input data that is suitable for this
        network."""
        # TODO: include blockpermute here
        return scipy.array(range(self.sequenceLength))

    def _buildTopology(self):
        inlayer = self.inlayerclass(self.indim)
        outlayer = self.outlayerclass(self.sequenceLength * self.outsize)
        self.hiddenlayers = []
        # Add connections and layers
        self.addInputModule(inlayer)
        for p in self._permsForSwiping():
            i = self._makeMdrnnLayer()
            self.hiddenlayers.append(i)
            # Make a connection that permutes the input...
            in_pc = PermutationConnection(inlayer, i, p, self.blocksize)
            # .. and one that permutes it back.
            pinv = permute(range(len(p)), p)
            out_pc = PermutationConnection(i, outlayer, pinv, self.outsize)
            self.addModule(i)
            self.addConnection(in_pc)
            self.addConnection(out_pc)
        self.addOutputModule(outlayer)

    def _permsForSwiping(self):
        """Return the correct permutations of blocks for all swiping direction.
        """
        # We use an identity permutation to generate the permutations from by
        # slicing correctly.
        return [self._standardPermutation()]

    def activate(self, inpt):
        inpt.shape = self.shape
        inpt_ = permuteToBlocks(inpt, self.blockshape)
        inpt.shape = scipy.size(inpt),
        return super(_Mdrnn, self).activate(inpt_)

    def filterResult(self, inpt):
        return inpt


class _MultiDirectionalMdrnn(_Mdrnn):

    def _permsForSwiping(self):
        """Return the correct permutations of blocks for all swiping direction.
        """
        # We use an identity permutation to generate the permutations from by
        # slicing correctly.
        identity = scipy.array(range(self.sequenceLength))
        identity.shape = tuple(s / b for s, b in zip(self.shape, self.blockshape))
        permutations = []
        # Loop over all possible directions: from each corner to each corner
        for direction in crossproduct([('+', '-')] * self.timedim):
            axises = []
            for _, axisdir in enumerate(direction):
                # Use a normal complete slice for forward...
                if axisdir == '+':
                    indices = slice(None, None, 1)
                # ...and a reversed complete slice for backward
                else:
                    indices = slice(None, None, -1)
                axises.append(indices)
            permutations.append(operator.getitem(identity, axises).flatten())
        return permutations


class _AccumulatingMdrnn(_Mdrnn):

    def activate(self, inpt):
        res = super(_AccumulatingMdrnn, self).activate(inpt)
        res.shape = self.outsize, self.indim
        res = res.sum()


