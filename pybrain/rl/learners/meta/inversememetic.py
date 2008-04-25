__author__ = 'Tom Schaul, tom@idsia.ch'

from memetic import MemeticSearch


class InverseMemeticSearch(MemeticSearch):
    """ Interleaving local search with topology search (inverse of memetic search) """
        
    def switchMutations(self):
        """ interchange the mutate() and topologyMutate() operators """
        tm = self.bestEvaluable.__class__.topologyMutate
        m = self.bestEvaluable.__class__.mutate
        self.bestEvaluable.__class__.topologyMutate = m
        self.bestEvaluable.__class__.mutate = tm
        
    def _learnStep(self):
        self.switchMutations()
        MemeticSearch._learnStep(self)
        self.switchMutations()