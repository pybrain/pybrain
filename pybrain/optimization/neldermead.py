__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy.optimize import fmin

from pybrain.optimization.optimizer import ContinuousOptimizer


class DesiredFoundException(Exception):
    """ The desired target has been found. """


class NelderMead(ContinuousOptimizer):
    """Do the optimization using a simple wrapper for scipy's fmin."""

    # acceptable relative error in the evaluator for convergence.
    stopPrecision = 1e-6

    mustMinimize = True


    def _callback(self, *_):
        if self._stoppingCriterion():
            raise DesiredFoundException()

    def _learnStep(self):
        try:
            fmin(func = self._oneEvaluation,
                 x0 = self.bestEvaluable,
                 callback = self._callback,
                 ftol = self.stopPrecision,
                 maxfun = self.maxEvaluations-self.numEvaluations-1,
                 disp = self.verbose)
        except DesiredFoundException:
            pass
        # the algorithm has finished: no point in doing more steps.
        self.maxLearningSteps = self.numLearningSteps
