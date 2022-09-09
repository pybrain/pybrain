__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


from pybrain.structure.connections.connection import Connection
from pybrain.structure.parametercontainer import ParameterContainer


class LinearConnection(Connection, ParameterContainer):
    """Connection that just forwards by multiplying the output of the inmodule
    with a parameter and adds it to the input of the outmodule."""

    def __init__(self, inmod, outmod, name=None,
                 inSliceFrom=0, inSliceTo=None, outSliceFrom=0, outSliceTo=None):
        if inSliceTo is None:
            inSliceTo = inmod.outdim
        size = inSliceTo - inSliceFrom
        Connection.__init__(self, inmod, outmod, name,
                            inSliceFrom, inSliceTo, outSliceFrom, outSliceTo)
        ParameterContainer.__init__(self, size)

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf += inbuf * self.params

    def _backwardImplementation(self, outerr, inerr, inbuf):
        #CHECKME: not setting derivatives -- this means the multiplicative weight is never updated!
        inerr += outerr * self.params
