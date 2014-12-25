__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.structure.connections.connection import Connection


class IdentityConnection(Connection):
    """Connection which connects the i'th element from the first module's output
    buffer to the i'th element of the second module's input buffer."""

    def __init__(self, *args, **kwargs):
        Connection.__init__(self, *args, **kwargs)
        assert self.indim == self.outdim, \
               "Indim (%i) does not equal outdim (%i)" % (
               self.indim, self.outdim)

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf += inbuf

    def _backwardImplementation(self, outerr, inerr, inbuf):
        inerr += outerr