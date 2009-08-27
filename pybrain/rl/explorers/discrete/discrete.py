__author__ = "Thomas Rueckstiess, ruecksti@in.tum.de"

from pybrain.rl.explorers.explorer import Explorer


class DiscreteExplorer(Explorer):
    """ discrete explorers choose one of the available actions from the
        set of actions. In order to know which actions are available and
        which action to choose, discrete explorers need access to the 
        module (which has to of class ActionValueTable).
    """
    
    _module = None
    
    def _setModule(self, module):
        """ tell the explorer the module (has to be ActionValueTable). """
        assert isinstance(module, ActionValueTable)
        self._module = module
    
    def _getModule(self):
        return self._module
    
    module = property(_getModule, _setModule)