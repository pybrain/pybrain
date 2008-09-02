__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.structure.evolvables.evolvable import Evolvable


class IncrementableComplexity(Evolvable):
    """ Subclasses of this have a concept of complexity of their parameter space,
    and that complexity can be increased with NO disturbance on their behavior,
    but after which the evolution can act in a larger search space. """
    
    def __init__(self, module, maxComplexity = None, **args):
        """ the default maxComplexity is the number of parameters in the module. """
        Evolvable.__init__(self, module)
        if maxComplexity == None:
            self.maxComplexity = self.module.paramdim
        else:
            self.maxComplexity = maxComplexity
        self.randomize(**args)
    
    def doubleMaxComplexity(self):
        self.maxComplexity *= 2
    
    def incrementMaxComplexity(self):
        self.maxComplexity += 1
    
    def newSimilarInstance(self):
        """ generate a new Evolvable with the same maximal complexity """
        res = self.copy()
        res.randomize()
        return res