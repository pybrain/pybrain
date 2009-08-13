__author__ = 'Tom Schaul, tom@idsia.ch'


from pybrain.optimization.optimizer import BlackBoxOptimizer
from pybrain.optimization.hillclimber import HillClimber
from pybrain.structure.modules.module import Module
from pybrain.structure.evolvables.maskedparameters import MaskedParameters
from pybrain.structure.evolvables.maskedmodule import MaskedModule
from pybrain.structure.evolvables.topology import TopologyEvolvable


class MemeticSearch(HillClimber):
    """ Interleaving topology search with local search """
    
    localSteps = 50
    localSearchArgs = {}
    localSearch = HillClimber
    
    def switchMutations(self):
        """ interchange the mutate() and topologyMutate() operators """
        tm = self._initEvaluable.__class__.topologyMutate
        m = self._initEvaluable.__class__.mutate
        self._initEvaluable.__class__.topologyMutate = m
        self._initEvaluable.__class__.mutate = tm            
    
    def _setInitEvaluable(self, evaluable):
        BlackBoxOptimizer._setInitEvaluable(self, evaluable)
        
        # distinguish modules from parameter containers.
        if not isinstance(evaluable, TopologyEvolvable):
            if isinstance(evaluable, Module):
                self._initEvaluable = MaskedModule(self._initEvaluable)
            else:
                self._initEvaluable = MaskedParameters(self._initEvaluable, returnZeros = True)      
            
    def _oneEvaluation(self, evaluable):
        if self.numEvaluations == 0:
            return BlackBoxOptimizer._oneEvaluation(self, evaluable)
        else:
            self.switchMutations()
            outsourced = self.localSearch(lambda x: BlackBoxOptimizer._oneEvaluation(self, x), evaluable, 
                                          maxEvaluations = self.localSteps,
                                          desiredEvaluation = self.desiredEvaluation,
                                          **self.localSearchArgs)
            _, fitness = outsourced.learn()
            self.switchMutations()
            return fitness
            
    def _learnStep(self):
        self.switchMutations()
        HillClimber._learnStep(self)
        self.switchMutations()
        
    def _notify(self):
        HillClimber._notify(self)
        if self.verbose:
            print 'bits on in best mask', sum(self.bestEvaluable.mask),
            
    @property
    def batchSize(self):
        return self.localSteps
    