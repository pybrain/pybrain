__author__ = 'Tom Schaul, tom@idsia.ch'


from pybrain.optimization.optimizer import BlackBoxOptimizer
from pybrain.optimization.hillclimber import HillClimber
from pybrain.structure.modules.module import Module
from pybrain.structure.evolvables.maskedparameters import MaskedParameters
from pybrain.structure.evolvables.maskedmodule import MaskedModule
from pybrain.structure.evolvables.topology import TopologyEvolvable


class MemeticSearch(BlackBoxOptimizer):
    """ Interleaving topology search with local search """
    
    localSteps = 100
    localSearchArgs = {}
    
    def __init__(self, evaluator, evaluable, localSearch = HillClimber, **args):
        """ Memetic search """
        self.localSearch = localSearch
        if not isinstance(evaluable, TopologyEvolvable):
            if isinstance(evaluable, Module):
                evaluable = MaskedModule(evaluable)
            else:
                evaluable = MaskedParameters(evaluable)
        BlackBoxOptimizer.__init__(self, evaluator, evaluable, **args)
        
        
    def _learnStep(self):
        # TODO: noisy case
        # CHECKME: topology mutation after or before local search?
        
        # only run a batch after accumulation enough evaluation steps
        if self.steps % self.localSteps != 0:
            return
        
        # produce a topology mutation
        challenger = self.bestEvaluable.copy()
        challenger.topologyMutate()
        
        # do a bit of local search on the new topology
        outsourced = self.localSearch(self.__evaluator, challenger, maxEvaluations = self.localSteps,
                                      desiredEvaluation = self.desiredEvaluation,
                                      **self.localSearchArgs)
        challenger, challengerFitness = outsourced.learn()
        
        if challengerFitness >= self.bestEvaluation:
            # keep the new mask
            self.bestEvaluable = challenger.copy()
            self.bestEvaluation = challengerFitness
        
        if self.verbose:
            print 'bits on in new mask', sum(challenger.mask),
            print self.bestEvaluation, challengerFitness
            