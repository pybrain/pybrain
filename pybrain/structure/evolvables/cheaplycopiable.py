__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.structure.modules.module import Module
from pybrain import Network


class CheaplyCopiable(ParameterContainer, Module):
    """ a shallow version of a module, that it only copies/mutates the params, not the structure. """
    
    def __init__(self, module):
        self.__stored = module
        self._params = module.params.copy()
        self.paramdim = module.paramdim    
        self.name = module.name+'-COPY'
        
    def copy(self):
        cp = CheaplyCopiable(self.__stored)
        cp.__stored._params[:] = self._params
        return cp
            
    @property
    def derivs(self): 
        self.__stored.derivs
    
    @property
    def _derivs(self): 
        self.__stored.derivs        
    
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
    
    