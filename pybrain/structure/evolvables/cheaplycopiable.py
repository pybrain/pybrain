__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.structure.modules.module import Module


class CheaplyCopiable(ParameterContainer, Module):
    """ a shallow version of a module, that it only copies/mutates the params, not the structure. """

    def __init__(self, module):
        self.__stored = module
        self._params = module.params.copy()
        self.paramdim = module.paramdim
        self.name = module.name+'-COPY'
        self.indim = module.indim
        self.outdim = module.outdim

    def copy(self):
        self.__stored._params[:] = self._params
        cp = CheaplyCopiable(self.__stored)
        return cp

    def convertToFastNetwork(self):
        self.__stored._params[:] = self._params
        cp = CheaplyCopiable(self.__stored.convertToFastNetwork())
        return cp

    @property
    def derivs(self):
        return self.__stored.derivs

    @property
    def _derivs(self):
        return self.__stored.derivs

    @property
    def outputbuffer(self):
        return self.__stored.outputbuffer

    @property
    def inputerror(self):
        return self.__stored.inputerror


    def reset(self):
        self.__stored.reset()

    def _resetBuffers(self):
        self.__stored._resetBuffers()

    def forward(self, *args, **kwargs):
        self.__stored._params[:] = self._params
        return self.__stored.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        self.__stored._params[:] = self._params
        return self.__stored.backward(*args, **kwargs)

    def activate(self, *args, **kwargs):
        self.__stored._params[:] = self._params
        return self.__stored.activate(*args, **kwargs)

    def backActivate(self, *args, **kwargs):
        self.__stored._params[:] = self._params
        return self.__stored.backActivate(*args, **kwargs)

    def randomize(self, *args, **kwargs):
        ParameterContainer.randomize(self, *args, **kwargs)
        self.__stored._params[:] = self._params

    def mutate(self, *args, **kwargs):
        ParameterContainer.mutate(self, *args, **kwargs)
        self.__stored._params[:] = self._params

    def getBase(self):
        self.__stored._params[:] = self._params
        return self.__stored

    def resetDerivatives(self):
        self.__stored.resetDerivatives()

