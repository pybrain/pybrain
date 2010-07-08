# -*- coding: utf-8 -_*-

__author__ = 'Justin Bayer, bayer.justin@googlemail.com'
__version__ = '$Id$'


from scipy import array

from pybrain.structure.connections.connection import Connection
from pybrain.utilities import permute


class PermutationConnection(Connection):
    """Connection that permutes the input by a given permutation."""

    def __init__(self, inmod, outmod, permutation, blocksize, *args, **kwargs):
        Connection.__init__(self, inmod, outmod, *args, **kwargs)
        if self.indim != self.outdim:
            raise ValueError("Indim (%i) does not equal outdim (%i)" % (
               self.indim, self.outdim))
        if len(permutation) * blocksize != self.indim:
            raise ValueError(
                "Permutation has wrong size: should be %i but is %i." %(
                (self.indim / blocksize), len(permutation)))

        self.permutation = array(permutation)
        self.invpermutation = permute(range(len(permutation)), permutation)
        self.blocksize = blocksize

    def _forwardImplementation(self, inbuf, outbuf):
        inbuf = inbuf.reshape(self.indim / self.blocksize, self.blocksize)
        inbuf = permute(inbuf, self.permutation)
        inbuf.shape = self.indim,
        outbuf += inbuf

    def _backwardImplementation(self, outerr, inerr, inbuf):
        outerr = outerr.reshape(self.indim / self.blocksize, self.blocksize)
        outerr = permute(outerr, self.invpermutation)
        outerr.shape = self.indim,
        inerr += outerr
