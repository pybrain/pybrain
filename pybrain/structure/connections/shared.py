__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.structure.connections.connection import Connection
from pybrain.structure.connections.full import FullConnection
from pybrain.structure.connections.subsampling import SubsamplingConnection


class OwnershipViolation(Exception):
    """Exception raised when one attempts to write-access the parameters of the
    SharedConnection, instead of its mother."""
    pass


class MotherConnection(ParameterContainer):
    """The container for the shared parameters of connections (just a container
    with a constructor, actually)."""

    hasDerivatives = True
    nbparams = None

    def __init__(self, nbparams, **args):
        assert nbparams > 0
        ParameterContainer.__init__(self, nbparams, **args)
        self.setArgs(nbparams = self.paramdim)


class SharedConnection(Connection):
    """A shared connection can link different couples of modules, with a single
    set of parameters (encapsulated in a MotherConnection)."""

    #: pointer to MotherConnection
    mother = None

    def __init__(self, mother, *args, **kwargs):
        Connection.__init__(self, *args, **kwargs)
        self._replaceParamsByMother(mother)

    def _replaceParamsByMother(self, mother):
        self.setArgs(mother = mother)
        self.paramdim = self.mother.paramdim

    def initParams(self, *args): raise OwnershipViolation
    @property
    def params(self): return self.mother.params

    @property
    def derivs(self): return self.mother.derivs

    def _getName(self):
        return self.mother.name if self._name is None else self._name

    def _setName(self, newname):
        self._name = newname

    name = property(_getName, _setName)


class SharedFullConnection(SharedConnection, FullConnection):
    """Shared version of FullConnection."""

    def _forwardImplementation(self, inbuf, outbuf):
        FullConnection._forwardImplementation(self, inbuf, outbuf)

    def _backwardImplementation(self, outerr, inerr, inbuf):
        FullConnection._backwardImplementation(self, outerr, inerr, inbuf)


class SharedSubsamplingConnection(SharedConnection, SubsamplingConnection):
    """Shared version of SubsamplingConnection."""

    def __init__(self, mother, inmod, outmod, **kwargs):
        SubsamplingConnection.__init__(self, inmod, outmod, **kwargs)
        self._replaceParamsByMother(mother)

    def _forwardImplementation(self, inbuf, outbuf):
        SubsamplingConnection._forwardImplementation(self, inbuf, outbuf)

    def _backwardImplementation(self, outerr, inerr, inbuf):
        SubsamplingConnection._backwardImplementation(self, outerr, inerr, inbuf)

