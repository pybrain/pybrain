__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from scipy import zeros
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

        