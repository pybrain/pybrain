__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.optimization.optimizer import BlackBoxOptimizer

    
class HillClimber(BlackBoxOptimizer):
    """ The simplest kind of stochastic search: hill-climbing in the fitness landscape. """    

    evaluatorIsNoisy = False
    
    def _learnStep(self):
        """ generate a new evaluable by mutation, compare them, and keep the best. """
        # re-evaluate the current individual in case the evaluator is noisy
        if self.evaluatorIsNoisy:
            self.bestEvaluation = self._oneEvaluation(self.bestEvaluable)
        
        challenger = self.bestEvaluable.copy()
        challenger.mutate()
        self._oneEvaluation(challenger)
        
        
class StochasticHillClimber(HillClimber):
    """ Stochastic hill-climbing always moves to a better point, but may also 
    go to a worse point with a probability that decreases with increasing drop in fitness
    (and depends on a temperature parameter). """
    
    