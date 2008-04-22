__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.learners import Learner

    
class WeightGuessing(Learner):
    """ a.k.a. random search """
    
    def _learnStep(self):
        new = self.bestEvaluable.copy()
        new.randomize()
        neweval = self.evaluator(new)
        if neweval >= self.bestEvaluation:
            self.bestEvaluator = new
            self.bestEvaluation = neweval
        
    