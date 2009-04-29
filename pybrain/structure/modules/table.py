__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from scipy import random, argmax
from module import Module


class ActionValueTable(Module):

    def __init__(self, numStates, numActions, name=None):
        Module.__init__(self, 1, 1, name)
        self.numStates = numStates
        self.numActions = numActions
        self.values = random.random((numStates, numActions))
        
    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[0] = self.getMaxAction(inbuf[0])
    
    def updateValue(self, state, action, value):
        self.values[state, action] = value
    
    def getValue(self, state, action):
        return self.values[state, action]
    
    def getMaxAction(self, state):
        return argmax(self.values[state,:])
            
    def initialize(self, value=0.0):
        self.values[:,:] = value       
            
        
        
        
        
        