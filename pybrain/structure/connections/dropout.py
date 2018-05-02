__author__ = 'god of alb, xwl992365231@gmail.com'

import numpy as np
from pybrain.structure.connections.connection import Connection
from pybrain.structure.parametercontainer import ParameterContainer

class FullConnectionWithDropout(Connection, ParameterContainer):
    """Full connection with dropout, the drop out rate r is the rate to drop the output from lower layer
   """

    def __init__(self, *args, r=0.1,**kwargs):
        Connection.__init__(self, *args, **kwargs)
        ParameterContainer.__init__(self, self.indim*self.outdim)
        self.r=r
        self.dropoutMask=np.random.binomial(1,(1-r),(self.indim))
    def __repr__(self):
        params = {
            'class': self.__class__.__name__,
            'name': self.name,
            'inmod': self.inmod.name,
            'outmod': self.outmod.name,
            'R':self.r,
            'mask':str(self.dropoutMask)
        }
        return "<%(class)s '%(name)s': '%(inmod)s' -> '%(outmod)s'> with dropout rate %(R)s\n mask:%(mask)s\n" % params
    def _forwardImplementation(self, inbuf, outbuf):
        inbuf*=self.dropoutMask
        outbuf += np.dot(np.reshape(self.params, (self.outdim, self.indim)),inbuf)

    def refresh(self):
        self.dropoutMask=np.random.binomial(1,(1-self.r),(self.indim))
    def _backwardImplementation(self, outerr, inerr, inbuf):
        
        inerr += np.dot(np.reshape(self.params, (self.outdim, self.indim)).T, outerr)*(self.dropoutMask.T)
        ds = self.derivs
        ds += (np.outer(inbuf, outerr).T*self.dropoutMask).flatten()
        

    def whichBuffers(self, paramIndex):
        """Return the index of the input module's output buffer and
        the output module's input buffer for the given weight."""
        return paramIndex % self.inmod.outdim, paramIndex // self.inmod.outdim
