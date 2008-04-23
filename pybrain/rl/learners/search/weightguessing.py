__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.learners import Learner
from pybrain.structure.modules.module import Module
from pybrain.structure.evolvables.maskedparameters import MaskedParameters
from pybrain.structure.evolvables.maskedmodule import MaskedModule
from pybrain.structure.evolvables.topology import TopologyEvolvable

    
class WeightGuessing(Learner):
    """ a.k.a. random search """
    
    def _learnStep(self):
        new = self.bestEvaluable.copy()
        new.randomize()
        neweval = self.evaluator(new)
        if neweval >= self.bestEvaluation:
            self.bestEvaluator = new
            self.bestEvaluation = neweval
        
        
class WeightMaskGuessing(WeightGuessing):
    """ random search, with a random mask that disables weights """
    
    def __init__(self, evaluator, evaluable, **args):
        assert not isinstance(evaluable, TopologyEvolvable)
        if isinstance(evaluable, Module):
            evaluable = MaskedModule(evaluable)
        else:
            evaluable = MaskedParameters(evaluable)
        Learner.__init__(self, evaluator, evaluable, **args)    
        
    