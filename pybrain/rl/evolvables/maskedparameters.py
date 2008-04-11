__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import zeros, randn
from random import random, sample, gauss

from incrementablecomplexity import IncrementableComplexity


class MaskedParameters(IncrementableComplexity):
    """ A module with a binary mask that can disable (=zero) parameters.
    If no maximum is set, the mask can potentially have all parameters enabled. 
    The maxComplexity represents the number of allowed enabled parameters. """
    
    maskFlipProbability = 0.05
    mutationStdev = 0.1    
        
    def randomize(self, **args):
        """ an initial, random mask (with random params) 
        with as many parameters enabled as allowed"""
        self.mask = zeros(self.module.paramdim, dtype=bool)
        for i in sample(range(self.module.paramdim), self.maxComplexity):
            self.mask[i] = True
        self.maskableParams = randn(self.module.paramdim)
        self._applyMask()
            
    def mutate(self, mask = False, weights = True, **args):
        """ the mutation can take place on two levels. """
        if weights:
            self._weightMutate(**args)
        if mask:
            self._maskMutate(**args)
        self._applyMask()
        
    def _applyMask(self):
        """ apply the mask to the module. """
        self.module._setParameters(self.mask*self.maskableParams)        
        
    def _maskMutate(self):
        """ flips some bits on the mask 
        (but do not exceed the maximum of enabled parameters). """
        for i in range(self.module.paramdim):
            if random() < self.maskFlipProbability:
                self.mask[i] = not self.mask[i]
        tooMany = sum(self.mask) - self.maxComplexity
        for i in range(tooMany):
            while True:
                ind = int(random()*self.module.paramdim)
                if self.mask[ind]:
                    self.mask[ind] = False
                    break
                
    def _weightMutate(self, sigma = None):
        """ add some gaussian noise to all parameters."""
        if sigma == None:
            sigma = self.mutationStdev
        for i in range(self.module.paramdim):
            self.maskableParams[i] += gauss(0, sigma)
           