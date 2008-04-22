__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.learners import Learner

    
class HillClimber(Learner):
    """ The simplest kind of stochastic search: hill-climbing in the fitness landscape. """    
    
    noisy = False
    
    def _learnStep(self):
        """ generate a new evaluable by mutation, compare them, and keep the best. """
        if self.noisy:
            # re-evaluate the current individual in case the evaluator is noisy
            self.bestEvaluation = self.evaluator(self.bestEvaluable)
        cp = self.bestEvaluable.copy()
        cp.mutate()
        tmpF = self.evaluator(cp)
        if tmpF >= self.bestEvaluation:
            self.bestEvaluable, self.bestEvaluation = cp, tmpF
        