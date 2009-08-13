from pybrain.rl.environments.functions.transformations import oppositeFunction
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
            if isinstance(evaluable, MaskedParameters):
                evaluable.returnZeros = False
                x0 = evaluable.params
                evaluable.returnZeros = True            
                def f(x): 
                    evaluable._setParameters(x)
                    return BlackBoxOptimizer._oneEvaluation(self, evaluable)
            else:
                f = lambda x: BlackBoxOptimizer._oneEvaluation(self, x)
                x0 = evaluable
            outsourced = self.localSearch(f, x0, 
                                          maxEvaluations = self.localSteps, 
                                          desiredEvaluation = self.desiredEvaluation,
                                          **self.localSearchArgs)
            assert self.localSteps > outsourced.batchSize, 'localSteps too small ('+str(self.localSteps)+\
                                                '), because local search has a batch size of '+str(outsourced.batchSize)
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
            print '  Bits on in best mask:', sum(self.bestEvaluable.mask)
            
    @property
    def batchSize(self):
        return self.localSteps
    