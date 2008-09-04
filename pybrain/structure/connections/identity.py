__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.utilities import substitute
from connection import Connection


class IdentityConnection(Connection):
    """ a connection which fully connects every element from the first module's output buffer
    to the second module's input buffer. """
    
    def __init__(self, *args, **kwargs):
        Connection.__init__(self, *args, **kwargs)
        assert self.indim == self.outdim
        
    def _forwardImplementation(self, inbuf, outbuf):
        outbuf += inbuf
        
    def _backwardImplementation(self, outerr, inerr, inbuf):
        inerr += outerr