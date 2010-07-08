__author__ = 'Frank Sehnke, sehnke@in.tum.de, Tom Schaul'

from scipy import random

from fd import FiniteDifferences


class SimpleSPSA(FiniteDifferences):
    """ Simultaneous Perturbation Stochastic Approximation.

    This class uses SPSA in general, but uses the likelihood gradient and a simpler exploration decay.
    """

    epsilon = 2. #Initial value of exploration size
    gamma = 0.9995 #Exploration decay factor
    batchSize = 2

    def _additionalInit(self):
        self.baseline = None #Moving average baseline, used just for visualisation

    def perturbation(self):
        # generates a uniform difference vector with the given epsilon
        deltas = (random.randint(0, 2, self.numParameters) * 2 - 1) * self.epsilon
        # reduce epsilon by factor gamma
        # as another simplification we let the exploration just decay with gamma.
        # Is similar to the decreasing exploration in SPSA but simpler.
        self.epsilon *= self.gamma
        return deltas

    def _learnStep(self):
        """ calculates the gradient and executes a step in the direction
            of the gradient, scaled with a learning rate alpha. """
        deltas = self.perturbation()
        #reward of positive and negative perturbations
        reward1 = self._oneEvaluation(self.current + deltas)
        reward2 = self._oneEvaluation(self.current - deltas)

        self.mreward = (reward1 + reward2) / 2.
        if self.baseline is None:
            # first learning step
            self.baseline = self.mreward * 0.99
            fakt = 0.
        else:
            #calc the gradients
            if reward1 != reward2:
                #gradient estimate alla SPSA but with likelihood gradient and normalization (see also "update parameters")
                fakt = (reward1 - reward2) / (2.0 * self.bestEvaluation - reward1 - reward2)
            else:
                fakt = 0.0
        self.baseline = 0.9 * self.baseline + 0.1 * self.mreward #update baseline

        # update parameters
        # as a simplification we use alpha = alpha * epsilon**2 for decaying the stepsize instead of the usual use method from SPSA
        # resulting in the same update rule like for PGPE
        self.current = self.gd(fakt * self.epsilon * self.epsilon / deltas)


