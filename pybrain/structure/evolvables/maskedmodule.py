__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.structure.evolvables.maskedparameters import MaskedParameters
from pybrain.structure.modules.module import Module


class MaskedModule(MaskedParameters, Module):
    """ an extension of masked-parameters, that wraps a module, and forwards the functionality. """

    def reset(self):
        return self.pcontainer.reset()

    def _resetBuffers(self):
        return self.pcontainer._resetBuffers()

    def activate(self, *args, **kwargs):
        return self.pcontainer.activate(*args, **kwargs)

    def backActivate(self, *args, **kwargs):
        return self.pcontainer.backActivate(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.pcontainer.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        return self.pcontainer.backward(*args, **kwargs)

    def activateOnDataset(self, *args, **kwargs):
        return self.pcontainer.activateOnDataset(*args, **kwargs)






