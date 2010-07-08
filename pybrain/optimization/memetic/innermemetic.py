__author__ = 'Tom Schaul, tom@idsia.ch'

from memetic import MemeticSearch
from pybrain.optimization.populationbased.es import ES


class InnerMemeticSearch(ES, MemeticSearch):
    """ Population-based memetic search """

    mu = 5
    lambada = 5

    def _learnStep(self):
        self.switchMutations()
        ES._learnStep(self)
        self.switchMutations()

    @property
    def batchSize(self):
        if self.evaluatorIsNoisy:
            return (self.mu + self.lambada)*self.localSteps
        else:
            return self.lambada*self.localSteps
