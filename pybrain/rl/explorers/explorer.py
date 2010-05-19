__author__ = "Thomas Rueckstiess, ruecksti@in.tum.de"


from pybrain.structure.modules.module import Module


class Explorer(Module):
    """ An Explorer object is used in Agents, receives the current state
        and action (from the controller Module) and returns an explorative
        action that is executed instead the given action.
    """

    def activate(self, state, action):
        """ The super class commonly ignores the state and simply passes the
            action through the module. implement _forwardImplementation()
            in subclasses.
        """
        return Module.activate(self, action)
    
        
    def newEpisode(self):
        """ Inform the explorer about the start of a new episode. """
        pass