__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.optimization.optimizer import BlackBoxOptimizer, TopologyOptimizer


class RandomSearch(BlackBoxOptimizer):
    """ Every point is chosen randomly, independently of all previous ones. """

    def _additionalInit(self):
        self._oneEvaluation(self._initEvaluable)

    def _learnStep(self):
        new = self._initEvaluable.copy()
        new.randomize()
        self._oneEvaluation(new)


class WeightGuessing(RandomSearch):
    """ Just an Alias. """


class WeightMaskGuessing(WeightGuessing, TopologyOptimizer):
    """ random search, with a random mask that disables weights """
