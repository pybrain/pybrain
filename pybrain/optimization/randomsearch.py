__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.optimization.optimizer import BlackBoxOptimizer
from pybrain.structure.modules.module import Module
from pybrain.structure.evolvables.maskedparameters import MaskedParameters
from pybrain.structure.evolvables.maskedmodule import MaskedModule
from pybrain.structure.evolvables.topology import TopologyEvolvable

    
class RandomSearch(BlackBoxOptimizer):
    """ Every point is chosen randomly, independently of all previous ones. """    
    
    def _learnStep(self):
        new = self.initEvaluable.copy()
        new.randomize()
        self._oneEvaluation(new)        


class WeightGuessing(RandomSearch):
    """ Just an Alias. """
        
        
class WeightMaskGuessing(WeightGuessing):
    """ random search, with a random mask that disables weights """
    
    def __init__(self, evaluator, evaluable, **args):
        assert not isinstance(evaluable, TopologyEvolvable)
        if isinstance(evaluable, Module):
            evaluable = MaskedModule(evaluable)
        else:
            evaluable = MaskedParameters(evaluable)
        BlackBoxOptimizer.__init__(self, evaluator, evaluable, **args)    
        
    