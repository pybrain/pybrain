__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from scipy import random, argmax, zeros
from module import Module


class Table(Module):
    """ implements a simple 2D table with dimensions rows x columns,
        which is basically a wrapper for a numpy array.
    """
    
    def __init__(self, numRows, numColumns, name=None):
        """ initialize with the number of rows and columns. the table
            values are all set to zero.
        """
        Module.__init__(self, 2, 1, name)
        self.numRows = numRows
        self.numColumns = numColumns
        self.values = zeros((numRows, numColumns), float)

    def _forwardImplementation(self, inbuf, outbuf):
        """ takes two coordinates, row and column, and returns the
            value in the table.
        """
        outbuf[0] = self.values[inbuf[0], inbuf[1]]
        
    def updateValue(self, row, column, value):
        """ set the value at a certain location in the table. """
        self.values[row, column] = value

    def getValue(self, row, column):
        """ return the value at a certain location in the table. """
        return self.values[row, column]



class ActionValueTable(Table):
    """ A special table that is used for Value Estimation methods
        in Reinforcement Learning.
    """
    
    def __init__(self, numStates, numActions, name=None):
        Module.__init__(self, 1, 1, name)
        self.numRows = numStates
        self.numColumns = numActions
        self.values = random.random((numStates, numActions))
        
    def _forwardImplementation(self, inbuf, outbuf):
        """ takes a vector of length 1 (the state coordinate) and returns
            the action with the maximum value over all actions for this state.
        """
        outbuf[0] = self.getMaxAction(inbuf[0])
        
    def getMaxAction(self, state):
        """ returns the action with the maximal value for the given state. """
        return argmax(self.values[state,:])
            
    def initialize(self, value=0.0):
        """ initializes the whole table with the given value. """
        self.values[:,:] = value       
    
        
        