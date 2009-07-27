__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy.optimize import fmin

from pybrain.optimization.optimizer import ContinuousOptimizer


class NelderMead(ContinuousOptimizer):
    """Do the optimization using a simple wrapper for scipy's fmin."""
    
    # acceptable relative error in the evaluator for convergence.
    stopPrecision = 1e-6
    
    minimize = True
    
    online = False
        
    def _batchLearn(self, maxSteps = None):
        """ The only stopping criterion (apart form limiting the evaluations) is
        to set the desired function precision. """
        self.bestEvaluable = fmin(func = self.evaluator, x0 = self.x0, ftol = self.stopPrecision, 
                                  maxfun = maxSteps, disp = self.verbose)
        self.bestEvaluation = self.evaluator(self.bestEvaluable)
        return self.bestEvaluable, self.bestEvaluation
    