__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from scipy import argmax, array, r_, asarray
from pybrain.utilities import abstractMethod
from pybrain.structure.modules import Table, Module
from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import one_to_n


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
        ParameterContainer.__init__(self, numStates * numActions)
        self.numRows = numStates
        self.numColumns = numActions

    @property
    def numActions(self):
        return self.numColumns

    def _forwardImplementation(self, inbuf, outbuf):
        """ Take a vector of length 1 (the state coordinate) and return
            the action with the maximum value over all actions for this state.
        """
        outbuf[0] = self.getMaxAction(inbuf[0])

    def getMaxAction(self, state):
        """ Return the action with the maximal value for the given state. """
        return argmax(self.params.reshape(self.numRows, self.numColumns)[state, :].flatten())

    def getActionValues(self, state):
        return self.params.reshape(self.numRows, self.numColumns)[state, :].flatten()

    def initialize(self, value=0.0):
        """ Initialize the whole table with the given value. """
        self._params[:] = value       


class ActionValueNetwork(Module, ActionValueInterface):
    def __init__(self, dimState, numActions, name=None):
        Module.__init__(self, dimState, 1, name)
        self.network = buildNetwork(dimState + numActions, dimState + numActions, 1)
        self.numActions = numActions
    
    def _forwardImplementation(self, inbuf, outbuf):
        """ takes a vector of length 1 (the state coordinate) and return
            the action with the maximum value over all actions for this state.
        """
        outbuf[0] = self.getMaxAction(asarray(inbuf))

    def getMaxAction(self, state):
        """ Return the action with the maximal value for the given state. """
        return argmax(self.getActionValues(state))

    def getActionValues(self, state):
        """ Run forward activation for each of the actions and returns all values. """
        values = array([self.network.activate(r_[state, one_to_n(i, self.numActions)]) for i in range(self.numActions)])
        return values

