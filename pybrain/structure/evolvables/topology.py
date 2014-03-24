__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.utilities import abstractMethod
from pybrain.structure.evolvables.evolvable import Evolvable
from pybrain.structure.parametercontainer import ParameterContainer


class TopologyEvolvable(ParameterContainer):
    """ An evolvable object, with higher-level mutations,
    that change the topology (in the broadest sense).
    It contains an instance of ParameterContainer. """

    pcontainer = None

    def __init__(self, pcontainer, **args):
        self.setArgs(**args)
        self.pcontainer = pcontainer

    @property
    def params(self):
        return self.pcontainer.params

    def _setParameters(self, x):
        self.pcontainer._setParameters(x)

    def topologyMutate(self):
        abstractMethod()

    def newSimilarInstance(self):
        """ generate a new Evolvable with the same topology """
        res = self.copy()
        res.randomize()
        return res

    def copy(self):
        """ copy everything, except the pcontainer """
        # CHECKME: is this correct, or might it be misleading?
        tmp = self.pcontainer
        self.pcontainer = None
        cp = Evolvable.copy(self)
        cp.pcontainer = tmp
        self.pcontainer = tmp
        return cp