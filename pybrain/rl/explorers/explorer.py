__author__ = "Thomas Rueckstiess, ruecksti@in.tum.de"

from scipy import random

from pybrain.utilities import abstractMethod, Named
from pybrain.structure.modules.table import ActionValueTable


class Explorer(object):
    """ An Explorer object is used in Agents, receives the current state
        and action (from the controller Module) and returns an explorative
        action that is executed instead the given action.
    """

    def activate(self, state, action):
        """ takes state and action and returns an explorative action. """
        abstractMethod()
        


class DiscreteExplorer(Explorer):
    """ discrete explorers choose one of the available actions from the
        set of actions. In order to know which actions are available and
        which action to choose, discrete explorers need access to the 
        module (which has to of class ActionValueTable).
    """
    
    __module = None
    
    def _setModule(self, module):
        """ tell the explorer the module (has to be ActionValueTable). """
        assert isinstance(module, ActionValueTable)
        self.__module = module
    
    def _getModule(self):
        return self.__module
    
    module = property(_getModule, _setModule)    