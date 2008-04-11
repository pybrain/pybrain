# TODO: this file does not work on the curren version of pybrain

from pybrain.structure.modules.module import Module
from pybrain.structure.modules.neuronlayer import NeuronLayer
from pybrain.structure import Network, FullConnection
import pybrain.structure as ps
from inspect import isclass
from random import choice, random


def buildComplexNetwork(units = 20, neuronrange = 100, recurrent = True):
    N = Network()
    moduleclasses = filter(lambda c: isclass(c) and issubclass(c, Module), vars(ps).values())
    moduleclasses.remove(Network)
    modules = []
    # build some modules, randomly
    for dummy in range(units):        
        mc = choice(moduleclasses)
        if issubclass(mc, NeuronLayer):
            m = mc(choice(range(neuronrange)))
        else:
            m = mc()
        modules.append(m)
    # add them to the network, some as input ans some as output though
    assert units > 4    
    for m in modules[:4]:
        N.addInputModule(m)
    for m in modules[4:-4]:
        N.addModule(m)
    for m in modules[-4:]:
        N.addOutputModule(m)
    # add random connections
    for mindex, m in enumerate(modules[:-2]):
        dest = choice(range(mindex+1, units))
        c = FullConnection(m, modules[dest])
        if recurrent and random() > 0.6:
            N.addRecurrentConnection(c)
        else:
            N.addConnection(c)
                
    N.sortModules()
    return N
        
if __name__ == '__main__':
    print buildComplexNetwork()
        