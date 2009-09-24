__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from scipy import random, argmax
from pybrain.utilities import abstractMethod
from pybrain.structure.modules import Table, Module
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.oneofn import *

class ActionValueInterface(object):
    """ Interface for different ActionValue modules, like the
        ActionValueTable or the ActionValueNetwork. 
    """
   
    numActions = None

    def getMaxAction(self, state):
        abstractMethod()
    
    def getActionValues(self, state):
        abstractMethod()
        
        
class ActionValueTable(Table, ActionValueInterface):
    """ A special table that is used for Value Estimation methods
        in Reinforcement Learning.
    """

    def __init__(self, numStates, numActions, name=None):
        Module.__init__(self, 1, 1, name)
        self.numRows = numStates
        self.numColumns = numActions
        self.values = random.random((numStates, numActions))

    @property
    def numActions(self):
        return self.numColumns

    def _forwardImplementation(self, inbuf, outbuf):
        """ takes a vector of length 1 (the state coordinate) and returns
            the action with the maximum value over all actions for this state.
        """
        outbuf[0] = self.getMaxAction(inbuf[0])

    def getMaxAction(self, state):
        """ returns the action with the maximal value for the given state. """
        return argmax(self.values[state,:].flatten())

    def getActionValues(self, state):
        return self.values[state, :].flatten()

    def initialize(self, value=0.0):
        """ initializes the whole table with the given value. """
        self.values[:,:] = value       


class ActionValueNetwork(Module, ActionValueInterface):
    def __init__(self, dimState, numActions):
        self.module = buildNetwork(dimState + numActions, dimState + numActions, 1)
        self.numActions = numActions
    
    def _forwardImplementation(self, inbuf, outbuf):
        """ takes a vector of length 1 (the state coordinate) and returns
            the action with the maximum value over all actions for this state.
        """
        outbuf[0] = self.getMaxAction(inbuf[0])

    def getMaxAction(self, state):
        """ returns the action with the maximal value for the given state. """
        return argmax(self.getActionValues())

    def getActionValues(self, state):
        values = array([self.module.activate(c_[state, one_to_n(i)]) for i in range(self.numActions)])
        return values
