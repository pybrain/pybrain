__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de, Tom Schaul'

from scipy import ones, zeros, dot, ravel, random
from scipy.linalg import pinv

from pybrain.auxiliary import GradientDescent
from pybrain.optimization.optimizer import ContinuousOptimizer


class FiniteDifferences(ContinuousOptimizer):
    """ Basic finite difference method. """

    epsilon = 1.0
    gamma = 0.999
    batchSize = 10

    # gradient descent parameters
    learningRate = 0.1
    learningRateDecay = None
    momentum = 0.0
    rprop = False

    def _setInitEvaluable(self, evaluable):
        ContinuousOptimizer._setInitEvaluable(self, evaluable)
        self.current = self._initEvaluable
        self.gd = GradientDescent()
        self.gd.alpha = self.learningRate
        if self.learningRateDecay is not None:
            self.gd.alphadecay = self.learningRateDecay
        self.gd.momentum = self.momentum
        self.gd.rprop = self.rprop
        self.gd.init(self._initEvaluable)

    def perturbation(self):
        """ produce a parameter perturbation """
        deltas = random.uniform(-self.epsilon, self.epsilon, self.numParameters)
        # reduce epsilon by factor gamma
        self.epsilon *= self.gamma
        return deltas

    def _learnStep(self):
        """ calls the gradient calculation function and executes a step in direction
            of the gradient, scaled with a small learning rate alpha. """

        # initialize matrix D and vector R
        D = ones((self.batchSize, self.numParameters))
        R = zeros((self.batchSize, 1))

        # calculate the gradient with pseudo inverse
        for i in range(self.batchSize):
            deltas = self.perturbation()
            x = self.current + deltas
            D[i, :] = deltas
            R[i, :] = self._oneEvaluation(x)
        beta = dot(pinv(D), R)
        gradient = ravel(beta)

        # update the weights
        self.current = self.gd(gradient)

