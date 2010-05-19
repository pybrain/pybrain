__author__ = "Thomas Rueckstiess, ruecksti@in.tum.de"

from scipy import random, dot

from pybrain.structure.modules.module import Module
from pybrain.rl.explorers.explorer import Explorer
from pybrain.tools.functions import expln, explnPrime
from pybrain.structure.parametercontainer import ParameterContainer


class StateDependentExplorer(Explorer, ParameterContainer):
    """ A continuous explorer, that perturbs the resulting action with
        additive, normally distributed random noise. The exploration
        has parameter(s) sigma, which are related to the distribution's
        standard deviation. In order to allow for negative values of sigma,
        the real std. derivation is a transformation of sigma according
        to the expln() function (see pybrain.tools.functions).
    """

    def __init__(self, statedim, actiondim, sigma= -2.):
        Explorer.__init__(self, actiondim, actiondim)
        self.statedim = statedim
        self.actiondim = actiondim

        # initialize parameters to sigma
        ParameterContainer.__init__(self, actiondim, stdParams=0)
        self.sigma = [sigma] * actiondim

        # exploration matrix (linear function)
        self.explmatrix = random.normal(0., expln(self.sigma), (statedim, actiondim))

        # store last state
        self.state = None

    def _setSigma(self, sigma):
        """ Wrapper method to set the sigmas (the parameters of the module) to a
            certain value.
        """
        assert len(sigma) == self.actiondim
        self._params *= 0
        self._params += sigma

    def _getSigma(self):
        return self.params

    sigma = property(_getSigma, _setSigma)

    def newEpisode(self):
        """ Randomize the matrix values for exploration during one episode. """
        self.explmatrix = random.normal(0., expln(self.sigma), self.explmatrix.shape)

    def activate(self, state, action):
        """ The super class commonly ignores the state and simply passes the
            action through the module. implement _forwardImplementation()
            in subclasses.
        """
        self.state = state
        return Module.activate(self, action)

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = inbuf + dot(self.state, self.explmatrix)

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        expln_params = expln(self.params
                        ).reshape(len(outbuf), len(self.state))
        explnPrime_params = explnPrime(self.params
                        ).reshape(len(outbuf), len(self.state))

        idx = 0
        for j in xrange(len(outbuf)):
            sigma_subst2 = dot(self.state ** 2, expln_params[j, :]**2)
            for i in xrange(len(self.state)):
                self._derivs[idx] = ((outbuf[j] - inbuf[j]) ** 2 - sigma_subst2) / sigma_subst2 * \
                    self.state[i] ** 2 * expln_params[j, i] * explnPrime_params[j, i]
                # if self.autoalpha and sigma_subst2 != 0:
                # self._derivs[idx] /= sigma_subst2
                idx += 1
            inerr[j] = (outbuf[j] - inbuf[j])
            # if not self.autoalpha and sigma_subst2 != 0:
            #     inerr[j] /= sigma_subst2
        # auto-alpha
        # inerr /= expln_sigma**2
        # self._derivs /= expln_sigma**2


