__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import zeros
from random import random, randint

from precisionboundparameters import PrecisionBoundParameters

#TODO: memetic approach / innovation protection


class BoundTotalInformation(PrecisionBoundParameters):
    """ all the parameters together are encoded with a limited total number of bits
    An additional list of integers keeps track of how many bits are associated
    with every parameter. 
    The maxComplexity represents the total of allowed bits. """

    precisionMutationProb = 0.1
        
    def mutate(self, **args):
        self._precisionMutation()
        self._weightMutation()
        self._applyWeights()
            
    def randomize(self):
        self.binaryWeights = zeros(self.module.paramdim, dtype=int)
        self.weightLengths = zeros(self.module.paramdim, dtype=int)
        # start with the allowed bits already randomly distributed over all the parameters
        for dummy in range(self.maxComplexity):
            ind = randint(0, self.module.paramdim-1)
            if self.weightLengths[ind] < 30:
                # no value should have a precision over 30 bits
                self.weightLengths[ind] += 1
        # randomly initialize the weights that have some bits attributed to them.
        for w in range(self.module.paramdim):
            if self.weightLengths[w] > 0:
                largest = 2**self.weightLengths[w]
                self.binaryWeights[w] = randint(0, largest-1)
        self._applyWeights()
        
    def doubleMaxComplexity(self):
        self.maxComplexity *= 2
            
    def incrementMaxComplexity(self):        
        self.maxComplexity += 1
        
    def _applyWeights(self):
        tmp = zeros(self.module.paramdim)
        # scale it to the range [-range, range]
        for w in range(self.module.paramdim):
            if self.weightLengths[w] == 0:
                tmp[w] = 0
            else:
                largest = 2**self.weightLengths[w]        
                tmp[w] = self.weightRange - (2.*self.binaryWeights[w]*self.weightRange)/largest
        self.module._setParameters(tmp)
    
    def _weightMutation(self):
        for w in range(self.module.paramdim):
            if self.weightLengths[w] > 0 and random() < self.mutationProbability:
                self.binaryWeights[w] = self._mutateBinary(self.binaryWeights[w], self.weightLengths[w])
            
    def _precisionMutation(self):
        for w in range(self.module.paramdim):
            if self.weightLengths[w] < 30 and random() < self.precisionMutationProb:
                self.weightLengths[w] += 1
                self.binaryWeights[w] *= 2
            if self.weightLengths[w] > 0  and random() < self.precisionMutationProb:
                self.weightLengths[w] -= 1
                self.binaryWeights[w] /= 2
        
        # check total   
        tooMany = sum(self.weightLengths) - self.maxComplexity 
        for dummy in range(tooMany):
            while True:
                ind = randint(0, self.module.paramdim-1)
                if self.weightLengths[ind] > 0:
                    self.weightLengths[ind] -= 1
                    self.binaryWeights[ind] /= 2
                    break