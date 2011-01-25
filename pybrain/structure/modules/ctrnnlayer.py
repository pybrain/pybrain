__author__ = "Daniel L Elliott, danelliottster@gmail.com"

from neuronlayer import NeuronLayer
import numpy as np

class CTRNNLayer(NeuronLayer):
    """ Layer which merely computes the potential of a CTRNN node.  Activation and inputs should be handled in seperate layers. """

    def __init__(self,dim,name=None):
        NeuronLayer.__init__(self, dim, name)
        # initialize tau values to 1
        self._tauVals = np.ones(dim)
        # these are the derivatives of u with respect to the error
        self._deriv_t_plus_one = np.zeros(dim)
        # these are the u values used accross time steps
        self._u = np.zeros(dim)

    def setTau(self, tau):
        "blah blah blah"
        assert len(tau) == self.outdim   # outdim from Module class
        self._tauVals[:] = tau[:]

    def _forwardImplementation(self, inbuf, outbuf):
        "blah blah blah"

        self._u = np.add(np.multiply(1 - (1/self._tauVals),
                                     self._u),
                         np.multiply(1/self._tauVals,
                                     inbuf)
                         )

        outbuf[:] = self._u[:]

    def _backwardImplementation(self,outerr, inerr, outbuf, inbuf):
        "blah blah blah"

        self._deriv_t_plus_one = np.add(np.multiply(1 - (1/self._tauVals),
                                                    self._deriv_t_plus_one),
                                        outerr)
        
        inerr[:] = (1 / self._tauVals) * _deriv_t_plus_one
