#! /usr/bin/env python2.5
# -*- coding: utf-8 -*


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'

import ctypes

from itertools import chain


from pybrain.structure.networks.network import Network
from pybrain.structure.networks.feedforward import \
    FeedForwardNetworkComponent, FeedForwardNetwork
from pybrain.structure.networks.recurrent import RecurrentNetworkComponent, \
    RecurrentNetwork

from pybrain.structure.modules.neuronlayer import NeuronLayer

from arac.structure import c_parameter_container, c_bias_layer, \
    c_identity_layer, c_sigmoid_layer, c_lstm_layer, c_identity_connection, \
    c_full_connection, c_layer, c_connection
    
from ctypes import c_int, pointer
    

libarac = ctypes.CDLL('../libarac.so')     # This is like an import.


class _Network(Network):
    """A network that behaves exactly as a pybrain network, but uses 
    arac internally."""

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

    def addConnection(self, *args, **kwargs):
        Network.addConnection(self, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        super(_Network, self).__init__(*args, **kwargs)
        # These dictionaries are registries for already build structs.
        self.cmodules = {}
        self.cconnections = {}
        
        # These are used so the held modules don't have to be set the offset 
        # everytime it changes, but instead can use a pointer.
        self._coffset = c_int(0)
        self._cmaxoffset = c_int(0)
            
    def sortModules(self):
        super(_Network, self).sortModules()
        self.rebuild()
        
    def reset(self):
        super(_Network, self).sortModules()
        self.rebuild()
        
    def _growBuffers(self):
        self.growBuffersOnNetwork(self)
            
    def growBuffersOnNetwork(self, network):
        for module in network.modules:
            if isinstance(module, Network):
                self.growBuffersOnNetwork(module)
            else:
                module.offset = self.offset
                module.maxoffset = self.maxoffset
                module._growBuffers()
        
    def rebuild(self):
        self.buildCStructure()
        # The modules are sorted in the modules list in the right way. We create
        # an array with the corresponding cmodules in the same order.
        ordered_modules = [self.cmodules[m.name] for m in self.modulesSorted]
        c_modulearray = c_layer * len(self.cmodules)
        self.outmodule_buffer_pointer = self.outmodules[0].outputbuffer[0].ctypes.data
        self.modulearray = c_modulearray(*ordered_modules)
        self.mustRebuild = False

    def buildCStructure(self):
        """Build up a module-graph of c structs in memory."""
        # First make a struct for every module
        root = None
        for module in self.modules:
            if not isinstance(module, NeuronLayer):
                raise ValueError("Can only translate layermodules.")
            struct_module = self.buildCStructForLayer(module)
            self.cmodules[module.name] = struct_module
            root = root if root is not None else struct_module
            
        # Then we add connections. Mind that this has to modifiy the previous
        # structs.
        connection_iter = chain(*[cs for cs in self.connections.values()])
        for connection in connection_iter:
            struct_connection = self.buildCStructForConnection(connection)
            key = '%s-%i' % (connection.name, id(connection))
            self.cconnections[key] = struct_connection
        
        self.cstruct = root   
        
    def buildCStructForLayer(self, layer):
        struct = c_layer.from_layer(layer)
        # These are set here and not in the constructor, since they depend on 
        # the network that holds the layer.
        struct.timestep_p = pointer(self._coffset)
        struct.seqlen_p = pointer(self._cmaxoffset)
        return struct
        
    def buildCStructForConnection(self, connection):
        inlayer = self.cmodules[connection.inmod.name]
        outlayer = self.cmodules[connection.outmod.name]
        struct = c_connection.from_connection(connection, inlayer, outlayer)
        
        inlayer.add_outgoing_connection(struct)
        outlayer.add_incoming_connection(struct)

        return struct
        
    def activate(self, inputbuffer):
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

        # The outputbuffer of the first module in the list is the we decide upon
        # wether we have to grow the buffers.
        # This relies upon the fact, that all buffers of the network have the 
        # same size.
        indicating_buffer = self.inmodules[0].outputbuffer
        if self.offset >= indicating_buffer.shape[0]:
            self._growBuffers()
            self.rebuild()
            
        return [m.outputbuffer[self.offset - 1] for m in self.outmodules] 
        
    def backActivate(self, outerr):
        self.outputerror[self.offset - 1] = outerr
        libarac.calc_derivs(self.modulearray, len(self.modules))
        return self.inputerror[self.offset].copy()
        
        
class _FeedForwardNetwork(FeedForwardNetworkComponent, _Network):
    def __init__(self, *args, **kwargs):
        _Network.__init__(self, *args, **kwargs)        
        FeedForwardNetworkComponent.__init__(self, *args, **kwargs)
        
    # Arac always increments/decrements the offset after 
    # activation/backactivation. In the case of FFNs, we have to compensate for
    # this.
    
    def activate(self, inputbuffer):
        result = _Network.activate(self, inputbuffer)#
        self.offset -= 1
        return result
        
    def backActivate(self, outerr):
        self.offset += 1
        result = _Network.backActivate(self, outerr)
        return result
        
class _RecurrentNetwork(RecurrentNetworkComponent, _Network):
    def __init__(self, *args, **kwargs):
        _Network.__init__(self, *args, **kwargs)        
        RecurrentNetworkComponent.__init__(self, *args, **kwargs)
