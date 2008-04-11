__author__ = 'Tom Schaul, tom@idsia.ch'

from hillclimber import HillClimber
from pybrain.rl.evolvables import MaskedParameters


class MemeticHillClimber(HillClimber):
    """ A hillclimber for MaskedParameters: it acts on two different timescales
    for mask mutations and weight mutations. """
    
    localSteps = 50
    
    def __init__(self, evolvable, evaluator, localSteps = None, desiredFitness = None):
        """ @param localSteps: nb of weight mutations before a mask mutation happens. """
        assert isinstance(evolvable, MaskedParameters)
        self.desiredFitness = desiredFitness
        if localSteps != None:
            self.localSteps = localSteps
        HillClimber.__init__(self, evolvable, evaluator)
        self.challenger = self.evolvable.copy()
        self.challengerFitness = self.bestFitness
    
    def _oneStep(self, verbose = False):
        if self.steps % self.localSteps == 0:
            self.challengerFitness = self.evaluator(self.challenger)
            self.bestFitness = self.evaluator(self.evolvable)
            if self.challengerFitness >= self.bestFitness:
                # keep the new mask
                self.evolvable = self.challenger.copy()
                self.bestFitness = self.challengerFitness
            else:
                # throw the mask
                self.challenger = self.evolvable.copy()
                self.challenger.mutate(mask = True, weights = False)            
        else:
            self.challenger, self.challengerFitness = self._localSearchStep(self.challenger)
        if verbose:
            print sum(self.challenger.mask),
            print self.bestFitness, self.challengerFitness
