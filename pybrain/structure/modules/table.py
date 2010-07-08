__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'


from pybrain.structure.modules.module import Module
from pybrain.structure.parametercontainer import ParameterContainer

class Table(Module, ParameterContainer):
    """ implements a simple 2D table with dimensions rows x columns,
        which is basically a wrapper for a numpy array.
    """

    def __init__(self, numRows, numColumns, name=None):
        """ initialize with the number of rows and columns. the table
            values are all set to zero.
        """
        Module.__init__(self, 2, 1, name)
        ParameterContainer.__init__(self, numRows*numColumns)

        self.numRows = numRows
        self.numColumns = numColumns

    def _forwardImplementation(self, inbuf, outbuf):
        """ takes two coordinates, row and column, and returns the
            value in the table.
        """
        outbuf[0] = self.params.reshape(self.numRows, self.numColumns)[inbuf[0], inbuf[1]]

    def updateValue(self, row, column, value):
        """ set the value at a certain location in the table. """
        self.params.reshape(self.numRows, self.numColumns)[row, column] = value

    def getValue(self, row, column):
        """ return the value at a certain location in the table. """
        return self.params.reshape(self.numRows, self.numColumns)[row, column]

