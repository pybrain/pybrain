__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.optimization.optimizer import BlackBoxOptimizer, TabuOptimizer
from scipy import exp
from random import random


class HillClimber(BlackBoxOptimizer):
    """ The simplest kind of stochastic search: hill-climbing in the fitness landscape. """

    evaluatorIsNoisy = False

    def _additionalInit(self):
        self._oneEvaluation(self._initEvaluable)

    def _learnStep(self):
        """ generate a new evaluable by mutation, compare them, and keep the best. """
        # re-evaluate the current individual in case the evaluator is noisy
        if self.evaluatorIsNoisy:
            self.bestEvaluation = self._oneEvaluation(self.bestEvaluable)

        # hill-climbing
        challenger = self.bestEvaluable.copy()
        challenger.mutate()
        self._oneEvaluation(challenger)

    @property
    def batchSize(self):
        if self.evaluatorIsNoisy:
            return 2
        else:
            return 1

class TabuHillClimber(TabuOptimizer,HillClimber):
    """Applies the tabu proccess in addition to a hill climbing search."""

    def _learnStep(self):
        """generate a new a evaluable by mutation and check if it is tabu, repeat until a non-tabu
        evaluable is created then keep it and update the tabu list iff the new evaluable is an improvement"""
                                              
        if self.evaluatorIsNoisy:
            self.bestEvaluation = self._oneEvaluation(self.bestEvaluable)
        tabu=True
        old=self.bestEvaluable
        while tabu:
            challenger = self.bestEvaluable.copy()
            challenger.mutate()
            tabu=False
            for t in self.tabuList:
                if t(challenger):
                    tabu=True
        self._oneEvaluation(challenger)
        if all(challenger.params[x]==self.bestEvaluable.params[x] for x in range(0,len(challenger))):
            self.tabuList.append(self.tabuGenerator(old,self.bestEvaluable))
            l=len(self.tabuList)
            if l > self.maxTabuList:
                self.tabuList=self.tabuList[(l-self.maxTabuList):l]
   
class StochasticHillClimber(HillClimber):
    """ Stochastic hill-climbing always moves to a better point, but may also
    go to a worse point with a probability that decreases with increasing drop in fitness
    (and depends on a temperature parameter). """

    #: The larger the temperature, the more explorative (less greedy) it behaves.
    temperature = 1.

    def _learnStep(self):
        """re-evaluate the current individual in case the evaluator is noisy"""
        if self.evaluatorIsNoisy:
            self.bestEvaluation = self._oneEvaluation(self.bestEvaluable)

        # hill-climbing
        challenger = self.bestEvaluable.copy()
        challenger.mutate()
        newEval = self._oneEvaluation(challenger)

        # if the new evaluation was better, it got stored automatically. Otherwise:
        if ((not self.minimize and newEval < self.bestEvaluation) or
            (self.minimize and newEval > self.bestEvaluation)):
            acceptProbability = exp(-abs(newEval-self.bestEvaluation)/self.temperature)
            if random() < acceptProbability:
                self.bestEvaluable, self.bestEvaluation = challenger, newEval

