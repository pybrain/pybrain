#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-

"""This module provides a bridge from arac to PyBrain.

Arac features ways to compose networks similar to PyBrain. Since arac is C++ 
which is not easily usable from python, inline weave is used to construct an arac
network in parallel.

To ease this, this module features the classes _FeedForwardNetwork and
_RecurrentNetwork, which mimic the behaviour of the PyBrain API. Whenever you
want to use arac's speed benefits, use these classes instead of its pybrain
counterparts.
"""


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import copy

import scipy

from pybrain.structure import (
    BiasUnit,
    FeedForwardNetwork,
    FullConnection,
    GateLayer,
    IdentityConnection, 
    LinearConnection, 
    LinearLayer, 
    LSTMLayer,
    MDLSTMLayer,
    Network,
    PartialSoftmaxLayer,
    RecurrentNetwork,
    SharedFullConnection,
    SigmoidLayer, 
    SoftmaxLayer,
    TanhLayer,
)

from pybrain.structure.networks.feedforward import \
    FeedForwardNetworkComponent, FeedForwardNetwork
from pybrain.structure.networks.recurrent import RecurrentNetworkComponent, \
    RecurrentNetwork
    
import arac.cppbridge as cppbridge


class PybrainAracMapper(object):
    """Class that holds pybrain objects mapped to arac objects and provides 
    handlers to create new ones from pybrain objects."""
    
    classmapping = {
        BiasUnit: cppbridge.Bias,
        LinearLayer: cppbridge.LinearLayer, 
        GateLayer: cppbridge.GateLayer, 
        LSTMLayer: cppbridge.LstmLayer,
        SigmoidLayer: cppbridge.SigmoidLayer, 
        SoftmaxLayer: cppbridge.SoftmaxLayer,
        TanhLayer: cppbridge.TanhLayer,
        IdentityConnection: cppbridge.IdentityConnection, 
        FullConnection: cppbridge.FullConnection,
        SharedFullConnection: cppbridge.FullConnection,
        LinearConnection: cppbridge.LinearConnection,
    }

    def __init__(self):
        self.clear()
    
    def __del__(self):
        self.clear()
        
    def __getitem__(self, key):
        return self.map[key]
        
    def __setitem__(self, key, value):
        self.map[key] = value
    
    def __contains__(self, key):
        return key in self.map
    
    def clear(self):
        """Free the current map and all the held structures."""
        self.map = {}

    def _network_handler(self, network):
        # See if there already is a proxy:
        try: 
            proxy = self[network]
        except KeyError:
            proxy = cppbridge.Network()
            self[network] = proxy
        proxy.init_input(network.inputbuffer)
        proxy.init_output(network.outputbuffer)
        proxy.init_inerror(network.inputerror)
        proxy.init_outerror(network.outputerror)
        return proxy

    def _simple_layer_handler(self, layer):
        try:
            proxy = self[layer]
        except KeyError:
            proxy = self.classmapping[layer.__class__](layer.outdim)
            self[layer] = proxy
        proxy.init_input(layer.inputbuffer)
        proxy.init_output(layer.outputbuffer)
        proxy.init_inerror(layer.inputerror)
        proxy.init_outerror(layer.outputerror)
        return proxy
        
    def _bias_handler(self, bias):
        try:
            proxy = self[bias]
        except KeyError:
            proxy = cppbridge.Bias()
            self[bias] = proxy
        proxy.init_input(bias.inputbuffer)
        proxy.init_output(bias.outputbuffer)
        proxy.init_inerror(bias.inputerror)
        proxy.init_outerror(bias.outputerror)
        return proxy
        
    def _lstm_handler(self, layer):
        # See if there already is a proxy:
        try: 
            proxy = self.map[layer]
        except KeyError:
            proxy = cppbridge.LstmLayer(layer.dim)
            self[layer] = proxy
        proxy.init_input(layer.inputbuffer)
        proxy.init_output(layer.outputbuffer)
        proxy.init_state(layer.state)
        proxy.init_inerror(layer.inputerror)
        proxy.init_outerror(layer.outputerror)
        proxy.init_state_error(layer.stateError)
        return proxy
        
    def _mdlstm_handler(self, layer):
        # See if there already is a proxy:
        try: 
            proxy = self.map[layer]
        except KeyError:
            proxy = cppbridge.MdlstmLayer(layer.dimensions, layer.dim)
            self[layer] = proxy
        proxy.init_input(layer.inputbuffer)
        proxy.init_output(layer.outputbuffer)
        proxy.init_inerror(layer.inputerror)
        proxy.init_outerror(layer.outputerror)
        
        # FIXME: we have to make a buffer especially for this layer to give to 
        # arac. We attach this to the original pybrain layer object.
        layer.inputx = scipy.zeros((layer.dimensions, layer.dim))
        proxy.init_input_squashed(layer.inputx)
        proxy.init_input_gate_squashed(layer.ingate);
        proxy.init_input_gate_unsquashed(layer.ingatex);
        proxy.init_output_gate_squashed(layer.outgate);
        proxy.init_output_gate_unsquashed(layer.outgatex);
        proxy.init_forget_gate_unsquashed(layer.forgetgatex);
        proxy.init_forget_gate_squashed(layer.forgetgate);
        
        return proxy
        

    def _parametrized_connection_handler(self, con):
        try:
            incoming = self.map[con.inmod]
            outgoing = self.map[con.outmod]
        except KeyError, e:
            raise ValueError("Connection of unknown modules: %s" % e)
        try:
            proxy = self.map[con]
        except KeyError:
            klass = self.classmapping[type(con)]
            proxy = klass(incoming, outgoing, 
                          con.params, con.derivs,
                          con.inSliceFrom, con.inSliceTo,
                          con.outSliceFrom, con.outSliceTo)
            self.map[con] = proxy
        return proxy
            
    def _identity_connection_handler(self, con):
        try:
            incoming = self.map[con.inmod]
            outgoing = self.map[con.outmod]
        except KeyError, e:
            raise ValueError("Connection of unknown modules: %s" % e)
        try:
            proxy = self.map[con]
        except KeyError:
            proxy = cppbridge.IdentityConnection(incoming, outgoing, 
                                                 con.inSliceFrom, con.inSliceTo,
                                                 con.outSliceFrom, con.outSliceTo)
            self.map[con] = proxy
        return proxy
        
    def handle(self, obj):
        handlers = {
            BiasUnit: self._bias_handler,
            FeedForwardNetwork: self._network_handler,
            FullConnection: self._parametrized_connection_handler,
            GateLayer: self._simple_layer_handler, 
            IdentityConnection: self._identity_connection_handler, 
            LinearConnection: self._parametrized_connection_handler,
            LinearLayer: self._simple_layer_handler, 
            LSTMLayer: self._lstm_handler,
            MDLSTMLayer: self._mdlstm_handler,
            Network: self._network_handler,
            RecurrentNetwork: self._network_handler,
            SharedFullConnection: self._parametrized_connection_handler,
            SigmoidLayer: self._simple_layer_handler, 
            SoftmaxLayer: self._simple_layer_handler,
            TanhLayer: self._simple_layer_handler,
            _FeedForwardNetwork: self._network_handler,
            _RecurrentNetwork: self._network_handler,
        }
        self.map[obj] = handlers[type(obj)](obj)
        return self.map[obj]
    

class _Network(Network):
    """Adapter for the pybrain Network class.

    Currently, the only way to process input to the network is the .activate()
    method. The only way to propagate the error back is .backActivate().
    """

    @property
    def offset(self):
        return self.proxies[self].timestep()

    def __init__(self, *args, **kwargs):
        super(_Network, self).__init__(*args, **kwargs)
        # Mapping the components of the network to their proxies.
        self.proxies = PybrainAracMapper()

        # This is to keep references of arrays that shall not be collected.
        self._dontcollect = []
        
    def copy(self):
        old_proxies = self.proxies
        self.proxies = PybrainAracMapper()
        result = copy.deepcopy(self)
        self.proxies = old_proxies
        result.sortModules()
        return result

    def _growBuffers(self):
        super(_Network, self)._growBuffers()
        self._rebuild()
            
    def sortModules(self):
        super(_Network, self).sortModules()
        self._rebuild()

    def _rebuild(self):
        self.buildCStructure()
        
    def reset(self):
        net_proxy = self.proxies.handle(self)
        net_proxy.clear()
        # Empty references.
        self._dontcollect[:] = []

    def buildCStructure(self):
        """Build up a C++-network."""
        # We first add all the modules, since we have to know about them before
        # we can add connections.
        net_proxy = self.proxies.handle(self)
        for module in self.modules:
            add = not module in self.proxies
            mod_proxy = self.proxies.handle(module)
            if add:
                inpt = module in self.inmodules
                outpt = module in self.outmodules
                mode = {
                    (False, False): cppbridge.Network.Simple,
                    (True, False): cppbridge.Network.InputModule,
                    (False, True): cppbridge.Network.OutputModule,
                    (True, True): cppbridge.Network.InputOutputModule,
                }[inpt, outpt]
                net_proxy.add_module(mod_proxy, mode)
        for connectionlist in self.connections.values():
            for connection in connectionlist:
                add = not connection in self.proxies
                con_proxy = self.proxies.handle(connection)
                if add:
                    net_proxy.add_connection(con_proxy)
        
    def activate(self, inpt):
        inpt = scipy.asarray(inpt, dtype='float64')
        # We reshape here in order to make sure that the array has the correct
        # dimensions when passed to the Swig-Proxy.
        inpt.shape = self.indim,
        result = scipy.zeros(self.outdim)
        self.proxies[self].activate(inpt, result)
        self._dontcollect += [inpt, result]
        return result
        
    def backActivate(self, outerr):
        outerr = scipy.asarray(outerr, dtype='float64')
        # We reshape here in order to make sure that the array has the correct
        # dimensions when passed to the Swig-Proxy.
        outerr.shape = self.outdim,
        inerror = scipy.zeros(self.indim)
        self.proxies[self].back_activate(outerr, inerror)
        self._dontcollect += [outerr, inerror]
        return inerror

        
class _FeedForwardNetwork(FeedForwardNetworkComponent, _Network):
    """Pybrain adapter for an arac FeedForwardNetwork."""

    def __init__(self, *args, **kwargs):
        _Network.__init__(self, *args, **kwargs)        
        FeedForwardNetworkComponent.__init__(self, *args, **kwargs)
        
    def activate(self, inputbuffer):
        result = _Network.activate(self, inputbuffer)
        return result
        
    def backActivate(self, outerr):
        result = _Network.backActivate(self, outerr)
        return result

        
class _RecurrentNetwork(RecurrentNetworkComponent, _Network):
    """Pybrain adapter for an arac RecurrentNetwork."""

    def __init__(self, *args, **kwargs):
        _Network.__init__(self, *args, **kwargs)        
        RecurrentNetworkComponent.__init__(self, *args, **kwargs)

    def activate(self, inputbuffer):
        while True:
            # Grow buffers until they have the correct size.
            if self.offset < self.outputbuffer.shape[0]:
                break
            # TODO: _growBuffers() is called more than once.
            self._growBuffers()
        result = _Network.activate(self, inputbuffer)
        return result

    def backActivate(self, outerr):
        result = _Network.backActivate(self, outerr)
        return result

    def buildCStructure(self):
        """Build up a module-graph of c structs in memory."""
        _Network.buildCStructure(self)
        net_proxy = self.proxies.handle(self)
        net_proxy.set_mode(cppbridge.Component.Sequential)
        for connection in self.recurrentConns:
            add = not connection in self.proxies
            con_proxy = self.proxies.handle(connection)
            con_proxy.set_mode(cppbridge.Component.Sequential)
            con_proxy.set_recurrent(1)
            if add:
                net_proxy.add_connection(con_proxy)
        # FIXME: This is actually done more often than necessary, basically all 
        # recurrent connections are set to sequential already in the previous 
        # loop.
        for component in self.proxies.map.values():
            component.set_mode(cppbridge.Component.Sequential)