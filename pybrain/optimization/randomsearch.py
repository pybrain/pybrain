__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.optimization.optimizer import BlackBoxOptimizer
from pybrain.structure.modules.module import Module
from pybrain.structure.evolvables.maskedparameters import MaskedParameters
from pybrain.structure.evolvables.maskedmodule import MaskedModule
from pybrain.structure.evolvables.topology import TopologyEvolvable

    
class RandomSearch(BlackBoxOptimizer):
    """ Every point is chosen randomly, independently of all previous ones. """    
    
    def _learnStep(self):
        new = self._initEvaluable.copy()
        new.randomize()
        self._oneEvaluation(new)        


class WeightGuessing(RandomSearch):
    """ Just an Alias. """
        
        
class WeightMaskGuessing(WeightGuessing):
    """ random search, with a random mask that disables weights """
    
    def _setInitEvaluable(self, evaluable):
        if isinstance(evaluable, TopologyEvolvable):
            raise ValueError('Initial evaluable cannot inherit from TopologyEvaluable')
        BlackBoxOptimizer._setInitEvaluable(self, evaluable)
        
        # distinguish modules from parameter containers.
        if isinstance(evaluable, Module):
            self._initEvaluable = MaskedModule(self._initEvaluable)
        else:
            self._initEvaluable = MaskedParameters(self._initEvaluable, returnZeros = True)      
        