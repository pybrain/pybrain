__author__ = 'Tom Schaul, tom@idsia.ch'

from random import random, randint
from scipy import zeros

from incrementablecomplexity import IncrementableComplexity
from pybrain.utilities import int2gray, gray2int


class PrecisionBoundParameters(IncrementableComplexity):
    """ Every weight is encoded with a limited number of bits. 
    The maxComplexity represents the number of allowed bits per weight. """
    
    # how far from 0 can weights maximally be?
    weightRange = 2.
    
    mutationProbability = 0.1
    
    def mutate(self, **args):
        for w in range(self.module.paramdim):
            if random() < self.mutationProbability:
                self.binaryWeights[w] = self._mutateBinary(self.binaryWeights[w], self.maxComplexity)
        self._applyWeights()
            
    def doubleMaxComplexity(self):
        self.maxComplexity += 1
        self.binaryWeights *= 2
            
    def incrementMaxComplexity(self):        
        raise Exception('Impossible operation: the complexity cannot increment by small steps, only double.')
            
    def _mutateBinary(self, i, size):
        """ as we have Gray Code, it's good enough to flip any one bit"""
        flipBit = randint(0, size-1)
        gray = int2gray(i)
        gray = gray ^ (2**flipBit)
        return gray2int(gray, size)
    
    def randomize(self):
        self.binaryWeights = zeros(self.module.paramdim, dtype=int)
        largest = 2**self.maxComplexity
        for w in range(self.module.paramdim):
            self.binaryWeights[w] = randint(0, largest-1)
        self._applyWeights()
        
    def _applyWeights(self):
        tmp = zeros(self.module.paramdim)
        # scale it to the range [-range, range]
        largest = 2**self.maxComplexity
        for w in range(self.module.paramdim):
            tmp[w] = self.weightRange - (2.*self.binaryWeights[w]*self.weightRange)/largest
        self.module._setParameters(tmp)
        
    