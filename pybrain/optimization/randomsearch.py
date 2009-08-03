__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import array, ndarray

from pybrain.optimization.optimizer import BlackBoxOptimizer
from pybrain.structure.modules.module import Module
from pybrain.structure.evolvables.maskedparameters import MaskedParameters
from pybrain.structure.evolvables.maskedmodule import MaskedModule
from pybrain.structure.evolvables.topology import TopologyEvolvable
from pybrain.structure.parametercontainer import ParameterContainer

    
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
        # If the evaluable is provided as a list of numbers or as an array,
        # we wrap it into a ParameterContainer.
        if isinstance(evaluable, list):
            evaluable = array(evaluable)
        if isinstance(evaluable, ndarray):            
            pc = ParameterContainer(len(evaluable))
            pc._setParameters(evaluable)
            self.wasWrapped = True
            evaluable = pc
        elif evaluable is None:
            raise ValueError('An initial evaluable must be specified to start optimization.')
        elif isinstance(evaluable, TopologyEvolvable):
            raise ValueError('Initial evaluable cannot inherit from TopologyEvaluable')
        
        # distinguish modules from parameter containers.
        if isinstance(evaluable, Module):
            self._initEvaluable = MaskedModule(evaluable)
        else:
            self._initEvaluable = MaskedParameters(evaluable)            
        self._oneEvaluation(self._initEvaluable)
        