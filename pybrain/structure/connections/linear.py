#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'
__version__ = '$Id$'


from pybrain.structure.connections.connection import Connection
from pybrain.structure.parametercontainer import ParameterContainer


class LinearConnection(Connection, ParameterContainer):
    """Connection that just forwards by multiplying the output of the inmodule 
    with a parameter and adds it to the input of the outmodule."""
    
    def __init__(self, inmod, outmod, *args, **kwargs):
        if inmod.dim != outmod.dim:
            raise ValueError(
                "LinearConnections only work between equally sized modules.")
        Connection.__init__(self, inmod, outmod, *args, **kwargs)
        ParameterContainer.__init__(self, inmod.dim)
        
    def _forwardImplementation(self, inbuf, outbuf):
        outbuf += inbuf * self.params
        
    def _backwardImplementation(self, outerr, inerr, inbuf):
        inerr += outerr * self.params