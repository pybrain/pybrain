__author__ = 'Tom Schaul, tom@idsia.ch'

from searchprocess import SearchProcess

    
class HillClimber(SearchProcess):
    """ The simplest kind of stochastic search: hill-climbing in the fitness landscape. """    
    
    noisy = True
    
    def _oneStep(self, verbose = False):
        self.evolvable, self.bestFitness = self._localSearchStep(self.evolvable)            
        if verbose:
            print self.bestFitness
    
    def _localSearchStep(self, old):
        """ re-evaluate the current individual (this is for the case where the evaluator is noisy), 
        generate a new one, compare them, and keep the best. """
        if self.noisy:
            oldfitness = self.evaluator(old)
        else:
            oldfitness = self.bestFitness
        cp = old.copy()
        cp.mutate()
        tmpF = self.evaluator(cp)
        if tmpF >= oldfitness:
            return cp, tmpF
        else:
            return old, oldfitness