__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.optimization.optimizer import BlackBoxOptimizer, TopologyOptimizer
from pybrain.optimization.hillclimber import HillClimber
from pybrain.structure.evolvables.maskedparameters import MaskedParameters


class MemeticSearch(HillClimber, TopologyOptimizer):
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
                                          minimize = self.minimize,
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
            print('  Bits on in best mask:', sum(self.bestEvaluable.mask))

    @property
    def batchSize(self):
        return self.localSteps
