__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from scipy import reshape, dot, outer, eye
from pybrain.structure.connections import FullConnection


class FullNotSelfConnection(FullConnection):
    """Connection which connects every element from the first module's
    output buffer to the second module's input buffer in a matrix multiplicative
    manner, EXCEPT the corresponding elements with the same index of each buffer
    (the diagonal of the parameter matrix is 0). Asserts that in and out dimensions
    are equal. """
    #:TODO: the values on the diagonal are counted as parameters but not used! FIX!

    def __init__(self, *args, **kwargs):
        FullConnection.__init__(self, *args, **kwargs)
        assert self.indim == self.outdim, \
            "Indim (%i) does not equal outdim (%i)" % (
            self.indim, self.outdim)

    def _forwardImplementation(self, inbuf, outbuf):
        p = reshape(self.params, (self.outdim, self.indim)) * (1-eye(self.outdim))
        outbuf += dot(p, inbuf)

    def _backwardImplementation(self, outerr, inerr, inbuf):
        p = reshape(self.params, (self.outdim, self.indim)) * (1-eye(self.outdim))
        inerr += dot(p.T, outerr)
        ds = self.derivs
        ds += outer(inbuf, outerr).T.flatten()
