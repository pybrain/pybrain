__author__ = "Thomas Rueckstiess, ruecksti@in.tum.de"

from scipy import random

from pybrain.rl.explorers.explorer import Explorer
from pybrain.tools.functions import expln, explnPrime
from pybrain.structure.parametercontainer import ParameterContainer


class NormalExplorer(Explorer, ParameterContainer):
    """ A continuous explorer, that perturbs the resulting action with
        additive, normally distributed random noise. The exploration
        has parameter(s) sigma, which are related to the distribution's
        standard deviation. In order to allow for negative values of sigma,
        the real std. derivation is a transformation of sigma according
        to the expln() function (see pybrain.tools.functions).
    """

    def __init__(self, dim, sigma=0.):
        Explorer.__init__(self, dim, dim)
        self.dim = dim

        # initialize parameters to sigma
        ParameterContainer.__init__(self, dim, stdParams=0)
        self.sigma = [sigma] * dim

    def _setSigma(self, sigma):
        """ Wrapper method to set the sigmas (the parameters of the module) to a
            certain value.
        """
        assert len(sigma) == self.dim
        self._params *= 0
        self._params += sigma

    def _getSigma(self):
        return self.params

    sigma = property(_getSigma, _setSigma)

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = random.normal(inbuf, expln(self.sigma))

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        expln_sigma = expln(self.sigma)
        self._derivs += ((outbuf - inbuf) ** 2 - expln_sigma ** 2) / expln_sigma * explnPrime(self.sigma)
        inerr[:] = (outbuf - inbuf)

        # auto-alpha
        # inerr /= expln_sigma**2
        # self._derivs /= expln_sigma**2


