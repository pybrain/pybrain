__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.utilities import abstractMethod


class SearchProcess(object):
    """ a process that messes around with an Evolvable object, trying to increase its fitness, as
    given by an evaluator.  """
    
    desiredFitness = None
    
    def __init__(self, evolvable, evaluator, desiredFitness = None):
        self.desiredFitness = desiredFitness
        self.steps = 0
        self.evolvable = evolvable
        self.evaluator = evaluator
        self.bestFitness = self.evaluator(self.evolvable)
    
    def search(self, nbSteps, **args):
        """ continue searching where we left off, for a nb of steps. """    
        for dummy in range(nbSteps):
            self._oneStep(**args)
            self.steps += 1
            if self.desiredFitness != None and self.bestFitness > self.desiredFitness:
                break
            
    def _oneStep(self, **args):
        """ to be implemented by subclasses. """
        abstractMethod()
        
    def increaseMaxComplexity(self):
        """ increase the complexity of the evolvable module(s), while
        preserving as much as possible of the old solution """
        self.evolvable.incrementMaxComplexity()
        
    def newSimilarInstance(self):
        """ new search process starting at step 0, with copies of the current modules. """
        tmp = self.evolvable.newSimilarInstance()
        return self.__class__(tmp, self.evaluator)