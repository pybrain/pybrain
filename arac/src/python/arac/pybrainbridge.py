#! /usr/bin/env python2.5
# -*- coding: utf-8 -*

"""This module provides a bridge from arac to PyBrain.

The arac library does intentionally not feature any ways of composing networks 
in an easy way. Instead, PyBrain should be used for setting up the networks 
which should then be converted to arac structures.

To ease this, this module features the classes _FeedForwardNetwork and 
_RecurrentNetwork, which mimic the behaviour of the PyBrain API. Whenever you 
want to use arac's speed benefits, use these classes instead of its pybrain
counterparts.
"""


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import ctypes
import itertools

import scipy

from arac.structure import c_parameter_container, c_bias_layer, \
    c_identity_layer, c_sigmoid_layer, c_lstm_layer, c_identity_connection, \
    c_full_connection, c_layer, c_connection, c_double_p
from arac.lib import libarac
from pybrain.structure.networks.network import Network
from pybrain.structure.networks.feedforward import \
    FeedForwardNetworkComponent, FeedForwardNetwork
from pybrain.structure.networks.recurrent import RecurrentNetworkComponent, \
    RecurrentNetwork
from pybrain.structure.modules.neuronlayer import NeuronLayer

    


class _Network(Network):
    """Counterpart of the pybrain Network class.
    
    In order to use arac's functionality, this class is used to construct and
    administer the needed structs. Those structs are created with ctypes and are
    accessible via the .cmodules and .cconnections fields.
    
    Currently, the only way to process input to the network is the .activate()
    method. The only way to propagate the error back is .backActivate().
    """

    # TODO: make sure names of modules and connections are unique
    
    def _getOffset(self): 
        return self._coffset.value
    
    def _setOffset(self, value):
        self._coffset.value = value
            
    offset = property(_getOffset, _setOffset)
    
    def _getMaxoffset(self): 
        return self._cmaxoffset.value
    
    def _setMaxoffset(self, value):
        self._cmaxoffset.value = value
            
    maxoffset = property(_getMaxoffset, _setMaxoffset)

    def __init__(self, *args, **kwargs):
        super(_Network, self).__init__(*args, **kwargs)
        # These dictionaries are registries for already build structs.
        self.cmodules = {}
        self.cconnections = {}
        
        # These are used so the held modules don't have to be set the offset 
        # everytime it changes, but instead can use a pointer.
        self._coffset = ctypes.c_int(0)
        self._cmaxoffset = ctypes.c_int(0)

    def reset(self):
        self.offset = 0
        for m in self.modules:
            m.offset = 0
        libarac.resetAll(self.modulearray, len(self.modules))
            
    def sortModules(self):
        super(_Network, self).sortModules()
        self._rebuild()

    def _rebuild(self):
        self.buildCStructure()
        # The modules are sorted in the modules list in the right way. We create
        # an array with the corresponding cmodules in the same order.
        ordered_modules = [self.cmodules[m.name] for m in self.modulesSorted]
        c_modulearray = c_layer * len(self.cmodules)
        self.modulearray = c_modulearray(*ordered_modules)

    def buildCStructure(self):
        """Build up a module-graph of c structs in memory."""
        # First make a struct for every module.
        root = None
        for module in self.modules:
            if not isinstance(module, NeuronLayer):
                raise ValueError("Can only translate layermodules.")
            struct_module = self.buildCStructForLayer(module)
            self.cmodules[module.name] = struct_module
            root = root if root is not None else struct_module
            
        # Then we add connections. Mind that this has to modifiy the previous
        # structs.
        connection_iter = itertools.chain(*[cs for cs in self.connections.values()])
        for connection in connection_iter:
            struct_connection = self.buildCStructForConnection(connection)
            key = '%s-%i' % (connection.name, id(connection))
            self.cconnections[key] = struct_connection
        
        self.cstruct = root
        
    def buildCStructForLayer(self, layer):
        struct = c_layer.from_layer(layer)
        # These are set here and not in the constructor, since they depend on 
        # the network that holds the layer.
        struct.timestep_p = ctypes.pointer(self._coffset)
        struct.seqlen_p = ctypes.pointer(self._cmaxoffset)
        return struct
        
    def buildCStructForConnection(self, connection):
        inlayer = self.cmodules[connection.inmod.name]
        outlayer = self.cmodules[connection.outmod.name]
        struct = c_connection.from_connection(connection, inlayer, outlayer)
        
        inlayer.add_outgoing_connection(struct)
        outlayer.add_incoming_connection(struct)

        return struct
        
    def activate(self, inputbuffer):
        # The outputbuffer of the first module in the list is which we decide
        # upon wether we have to grow the buffers.
        # This relies upon the fact, that all buffers of the network have the
        # same size (in terms of timesteps).
        while True:
            # Grow buffers until they have the correct size, so possibly call
            # _growBuffers() more than once.
            if self.offset < self.outputbuffer.shape[0]:
                break
            self._growBuffers()
            self._rebuild()

        start = 0

        for inmodule in self.inmodules:
            end = start + inmodule.indim
            inmodule.inputbuffer[self.offset][:] = inputbuffer[start:end]
            start = end

        libarac.activate(self.modulearray, len(self.modules))

        # Don't increment offset here, since it will already have been done in 
        # libarac.

        self.maxoffset = \
            self.offset if self.offset > self.maxoffset else self.maxoffset

        outbuffers = [m.outputbuffer[self.offset - 1] for m in self.outmodules]
        out = scipy.concatenate(outbuffers)
        self.outputbuffer[self.offset - 1][:] = scipy.concatenate(outbuffers)
        return self.outputbuffer[self.offset - 1]
        
    def backActivate(self, outerr):
        # Function libarac.calc_derivs decrements the offset, so we have to
        # compensate for that here first.
        start = 0
        for outmodule in self.outmodules:
            end = start + outmodule.outdim
            outmodule.outputerror[self.offset - 1][:] = outerr[start:end]
            start = end

        libarac.calc_derivs(self.modulearray, len(self.modules))
        return self.inputerror[self.offset].copy()
        
        
class _FeedForwardNetwork(FeedForwardNetworkComponent, _Network):
    """Counterpart to pybrain's FeedForwardNetwork."""
    
    def __init__(self, *args, **kwargs):
        _Network.__init__(self, *args, **kwargs)        
        FeedForwardNetworkComponent.__init__(self, *args, **kwargs)
        
    # Arac always increments/decrements the offset after 
    # activation/backactivation. In the case of FFNs, we have to compensate for
    # this.
    
    def activate(self, inputbuffer):
        self.reset()
        result = _Network.activate(self, inputbuffer)
        self.offset -= 1
        return result
        
    def backActivate(self, outerr):
        self.offset += 1
        result = _Network.backActivate(self, outerr)
        return result
        
class _RecurrentNetwork(RecurrentNetworkComponent, _Network):
    """Counterpart to pybrain's RecurrentNetwork."""
    
    def __init__(self, *args, **kwargs):
        _Network.__init__(self, *args, **kwargs)        
        RecurrentNetworkComponent.__init__(self, *args, **kwargs)

    def activate(self, inputbuffer):
        result = _Network.activate(self, inputbuffer)
        return result

    def backActivate(self, outerr):
        result = _Network.backActivate(self, outerr)
        return result
        
    def buildCStructure(self):
        """Build up a module-graph of c structs in memory."""
        _Network.buildCStructure(self)
        for connection in self.recurrentConns:
            struct_connection = self.buildCStructForConnection(connection)
            struct_connection.recurrent = 1
            key = '%s-%i' % (connection.name, id(connection))
            self.cconnections[key] = struct_connection