__author__ = "Thomas Rueckstiess, ruecksti@in.tum.de"

from pybrain.rl.explorers.explorer import Explorer
# from pybrain.rl.learners.valuebased.interface import ActionValueInterface

class DiscreteExplorer(Explorer):
    """ Discrete explorers choose one of the available actions from the
        set of actions. In order to know which actions are available and
        which action to choose, discrete explorers need access to the
        module (which has to of class ActionValueTable).
    """

    _module = None

    def __init__(self):
        Explorer.__init__(self, 1, 1)

    def _setModule(self, module):
        """ Tells the explorer the module (which has to be ActionValueTable). """
        # removed: cause for circular import
        # assert isinstance(module, ActionValueInterface)
        self._module = module

    def _getModule(self):
        return self._module

    module = property(_getModule, _setModule)