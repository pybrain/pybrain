__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.structure.connections.connection import Connection
from pybrain.structure.parametercontainer import ParameterContainer
from scipy import average

#:TODO: backward pass

class SubsamplingConnection(Connection, ParameterContainer):
    """Connection that just averages all the inputs before forwarding."""

    def __init__(self, inmod, outmod, name=None,
                 inSliceFrom=0, inSliceTo=None, outSliceFrom=0, outSliceTo=None):
        if outSliceTo is None:
            outSliceTo = outmod.indim
        size = outSliceTo - outSliceFrom
        Connection.__init__(self, inmod, outmod, name,
                            inSliceFrom, inSliceTo, outSliceFrom, outSliceTo)
        ParameterContainer.__init__(self, size)

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf += average(inbuf) * self.params

    def _backwardImplementation(self, outerr, inerr, inbuf):
        raise NotImplementedError()

