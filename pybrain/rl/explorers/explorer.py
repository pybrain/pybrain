__author__ = "Thomas Rueckstiess, ruecksti@in.tum.de"

from scipy import random

from pybrain.utilities import abstractMethod, Named
from pybrain.structure.modules.module import Module


class Explorer(Module):
    """ An Explorer object is used in Agents, receives the current state
        and action (from the controller Module) and returns an explorative
        action that is executed instead the given action.
    """

    def activate(self, state, action):
        """ the super class ignores the state and simply passes the
            action through the module. implement _forwardImplementation()
            in subclasses.
        """
        return Module.activate(self, action)
        
