__author__ = 'Tom Schaul, tom@idsia.ch'

from searchprocess import SearchProcess

    
class RandomSearch(SearchProcess):
    """ The trivial baseline: random weight guessing. """    
    
    
    def _oneStep(self, verbose = False):
        """ re-evaluate the current individual (this is for the case where the evaluator is noisy), 
        generate a new one, compare them, and keep the best. """
        oldfitness = self.evaluator(self.evolvable)
        cp = self.evolvable.copy()
        cp.randomize()
        tmpF = self.evaluator(cp)
        if tmpF >= oldfitness:
            self.evolvable = cp
            self.bestFitness = tmpF
        if verbose:
            print self.steps, ':', self.bestFitness, tmpF
        