__author__ = 'Tom Schaul, tom@idsia.ch'

import copy

from pybrain import Network
from pybrain.utilities import abstractMethod
    
    
class Evolvable(object):
    """ The parent class for all evolvable modules, i.e. which implement
    a mutation and a copy operator. """   
    
    def __init__(self, module):        
        self.module = module   
        
    def reset(self): self.module.reset()        
    def activate(self, input): return self.module.activate(input)
    
    def mutate(self, **args):
        """ Vary some properties of the underlying module, so that it's behavior 
        changes, (but not too abruptly). """
        abstractMethod()
        
    def copy(self):
        """ by default a full deep copy - subclasses should implement something faster, if
        appropriate. """
        cp = copy.deepcopy(self)
        if isinstance(self.module, Network):
            cp.module._setParameters(self.module.getParameters().copy())        
        return cp
    
    def randomize(self):
        """ randomly set all variable parameters """
        abstractMethod()
        