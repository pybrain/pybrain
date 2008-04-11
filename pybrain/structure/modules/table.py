from scipy import zeros
from module import Module


class Table(Module):
    """ This module represents a (sparse) table. It uses a python dictionary to store
        the data. Placing the index in the input buffer and executing a forward
        pass returns the element (a scipy array) that was stored in the indexed 
        field. """
           
    def __init__(self, indim, outdim, name = None):
        Module.__init__(self, indim, outdim, name)    
        self.setArgs(indim = indim, outdim = outdim)
        self.data = {}
        
    def _forwardImplementation(self, inbuf, outbuf):
        try:
            outbuf[:] = self.data[tuple(inbuf)]
        except KeyError:
            outbuf[:] = zeros((1, self.outdim))
        
    def setField(self, index, value):
        self.data[index] = value
    
    def clear(self):
        self.data = {}
        