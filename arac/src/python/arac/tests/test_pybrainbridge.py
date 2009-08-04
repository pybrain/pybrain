#! /usr/bin/env python2.5
# -*- coding: utf-8 -*

"""Unittests for the arac.pybrainbridge module."""


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com' 


import unittest

import scipy

import arac.pybrainbridge as pybrainbridge #@UnresolvedImport

from arac.tests.common import TestCase #@UnresolvedImport

from pybrain.structure import (
    LinearLayer, 
    BiasUnit,
    SigmoidLayer, 
    GateLayer,
    TanhLayer,
    LSTMLayer,
    SoftmaxLayer,
    PartialSoftmaxLayer,
    MDLSTMLayer,
    IdentityConnection, 
    FullConnection,
    Network,
    RecurrentNetwork,
    FeedForwardNetwork,
    ModuleMesh,
    BorderSwipingNetwork
)


scipy.random.seed(0)




class NetworkTestCase(TestCase):

    runs = 10

    def sync(self, net1, net2):
        net1.sortModules()
        net2.sortModules()
        if hasattr(net1, 'params'):
            net1.params[:] = net2.params[:] = scipy.random.random(net1.params.shape)

    def equivalence_recurrent(self, net, _net):
        self.sync(net, _net)
        for i in xrange(self.runs):
            inpt = scipy.random.random(net.indim)
            pybrain_res = net.activate(inpt)
            arac_res = _net.activate(inpt)
            if (pybrain_res != arac_res).any():
                for module in net.modulesSorted:
                    for bn, _ in module.bufferlist:
                        print module.name, bn
                        print getattr(net[module.name], bn)
                        print getattr(_net[module.name], bn)
                        print "-" * 20
            self.assertArrayNear(pybrain_res, arac_res)

        for _ in xrange(self.runs):
            error = scipy.random.random(net.outdim)
            pybrain_res = net.backActivate(error)
            arac_res = _net.backActivate(error)

            # if (pybrain_res != arac_res).any():
            #     for module in net.modulesSorted:
            #         for bn, _ in module.bufferlist:
            #             buf =  getattr(net[module.name], bn)
            #             _buf =  getattr(_net[module.name], bn)
            #             if (buf == _buf).all():
            #                 continue
            #             print module.name, bn
            #             print (buf - _buf).max()
            #             # print buf 
            #             # print _buf
            #             print "-" * 20

            self.assertArrayNear(pybrain_res, arac_res)
            if hasattr(_net, '_derivs'):
                self.assertArrayNear(_net.derivs, net.derivs)
                
        net.reset()
        _net.reset()
        self.assert_((_net.inputbuffer == 0.).all())
        
        for _ in xrange(self.runs):
            inpt = scipy.random.random(net.indim)
            pybrain_res = net.activate(inpt)
            arac_res = _net.activate(inpt)
            self.assertArrayNear(pybrain_res, arac_res)

        for _ in xrange(self.runs):
            error = scipy.random.random(net.outdim)
            pybrain_res = net.backActivate(error)
            arac_res = _net.backActivate(error)
            self.assertArrayNear(pybrain_res, arac_res)
            if hasattr(_net, '_derivs'):
                self.assertArrayNear(_net.derivs, net.derivs)
                        
    def equivalence_feed_forward(self, net, _net):
        self.sync(net, _net)
        for _ in xrange(self.runs):
            inpt = scipy.random.random(net.indim)
            pybrain_res = net.activate(inpt)
            arac_res = _net.activate(inpt)
            self.assertArrayNear(pybrain_res, arac_res)
            error = scipy.random.random(net.outdim)
            pybrain_res = net.backActivate(error)
            arac_res = _net.backActivate(error)
            self.assertArrayNear(pybrain_res, arac_res)
            if hasattr(_net, '_derivs'):
                self.assertArrayNear(_net.derivs, net.derivs)


class TestNetworkEquivalence(NetworkTestCase):
    
    def two_layer_network(self, net):
        inlayer = SigmoidLayer(2, 'in')
        outlayer = LinearLayer(2, 'out')
        con = FullConnection(inlayer, outlayer)
        net.addInputModule(inlayer)
        net.addOutputModule(outlayer)
        net.addConnection(con)
        
    def rec_two_layer_network(self, net):
        inlayer = LinearLayer(2, 'in')
        outlayer = LinearLayer(2, 'out')
        con = IdentityConnection(inlayer, outlayer)
        rcon = IdentityConnection(inlayer, outlayer)
        net.addInputModule(inlayer)
        net.addOutputModule(outlayer)
        net.addConnection(con)
        net.addRecurrentConnection(rcon)
        
    def sliced_connection_network(self, net):
        inlayer = LinearLayer(2, 'in')
        outlayer = LinearLayer(2, 'out')
        con = IdentityConnection(inlayer, outlayer, 
                                 inSliceFrom=0, inSliceTo=1,
                                 outSliceFrom=1, outSliceTo=2,
                                 )
        con = IdentityConnection(inlayer, outlayer, 
                         inSliceFrom=1, inSliceTo=2,
                         outSliceFrom=0, outSliceTo=1,
                         )
        net.addInputModule(inlayer)
        net.addOutputModule(outlayer)
        net.addConnection(con)

    def lstm_network(self, net):
        i = LinearLayer(1, name='in')
        h = LSTMLayer(2, name='hidden')
        o = LinearLayer(1, name='out')
        b = BiasUnit(name='bias')
        net.addModule(b)
        net.addOutputModule(o)
        net.addInputModule(i)
        net.addModule(h)
        net.addConnection(FullConnection(i, h))
        net.addConnection(FullConnection(b, h))
        net.addRecurrentConnection(FullConnection(h, h))
        net.addConnection(FullConnection(h, o))
        
    def lstm_cell(self, net):
        inpt = LinearLayer(4, 'inpt')
        forgetgate = GateLayer(1, 'forgetgate')
        ingate = GateLayer(1, 'ingate')
        outgate = GateLayer(1, 'outgate')
        state = LinearLayer(1, 'state')
        
        in_to_fg = IdentityConnection(inpt, forgetgate, 
                                      inSliceFrom=0, inSliceTo=1,
                                      outSliceFrom=0, outSliceTo=1,
                                      name='in_to_fg')
        in_to_og = IdentityConnection(inpt, outgate, 
                                      inSliceFrom=1, inSliceTo=2,
                                      outSliceFrom=1, outSliceTo=2,
                                      name='in_to_og')
        in_to_ig = IdentityConnection(inpt, ingate,
                                      inSliceFrom=2, inSliceTo=4,
                                      outSliceFrom=0, outSliceTo=2,
                                      name='in_to_ig')
        fg_to_st = IdentityConnection(forgetgate, state,
                                      name='fg_to_st')
        st_to_fg = IdentityConnection(state, forgetgate, 
                                      outSliceFrom=1, outSliceTo=2,
                                      name='st_to_fg'
                                      )
        st_to_og = IdentityConnection(state, outgate,
                                     outSliceFrom=1, outSliceTo=2,
                                     name='st_to_og'
                                     )
        ig_to_st = IdentityConnection(ingate, state, name='ig_to_st')
        
        net.addInputModule(inpt)
        net.addModule(forgetgate)
        net.addModule(ingate)
        net.addModule(state)
        net.addOutputModule(outgate)
        
        net.addConnection(in_to_fg)
        net.addConnection(in_to_og)
        net.addConnection(in_to_ig)
        net.addConnection(fg_to_st)
        net.addRecurrentConnection(st_to_fg)
        net.addConnection(st_to_og)
        net.addConnection(ig_to_st)
        
    def weird_network(self, net):
        bias = BiasUnit(name='bias')
        inlayer = TanhLayer(1, name='input')
        outlayer = TanhLayer(1, name='output')
        gatelayer = GateLayer(1, name='gate')
        con1 = FullConnection(bias, gatelayer, outSliceFrom=0, outSliceTo=1)
        con2 = FullConnection(bias, gatelayer, outSliceFrom=1, outSliceTo=2)
        con3 = FullConnection(inlayer, gatelayer, outSliceFrom=0, outSliceTo=1)
        con4 = FullConnection(inlayer, gatelayer, outSliceFrom=1, outSliceTo=2)
        con5 = FullConnection(gatelayer, outlayer)
        net.addInputModule(inlayer)
        net.addModule(bias)
        net.addModule(gatelayer)
        net.addOutputModule(outlayer)
        net.addConnection(con1)
        net.addConnection(con2)
        net.addConnection(con3)
        net.addConnection(con4)
        net.addConnection(con5)
        
    def xor_network(self, net):
        net.addInputModule(LinearLayer(2, name='in'))
        net.addModule(BiasUnit(name='bias'))
        net.addModule(LinearLayer(3, name='hidden'))
        net.addOutputModule(LinearLayer(1, name='out'))
        net.addConnection(FullConnection(net['in'], net['hidden']))
        net.addConnection(FullConnection(net['bias'], net['hidden']))
        net.addConnection(FullConnection(net['hidden'], net['out']))
        
    def rec_three_layer_network(self, net):
        inlayer = LinearLayer(1, name='in')
        hiddenlayer = LinearLayer(1, name='hidden')
        outlayer = LinearLayer(1, name='out')
        con1 = FullConnection(inlayer, hiddenlayer)
        con2 = FullConnection(hiddenlayer, outlayer)
        con3 = FullConnection(hiddenlayer, hiddenlayer)
        net.addInputModule(inlayer)
        net.addModule(hiddenlayer)
        net.addOutputModule(outlayer)
        net.addConnection(con1)
        net.addConnection(con2)
        net.addRecurrentConnection(con3)
        
    def equivalence_feed_forward(self, builder):
        _net = pybrainbridge._FeedForwardNetwork()
        builder(_net)
        net = FeedForwardNetwork()
        builder(net)
        super(TestNetworkEquivalence, self).equivalence_feed_forward(net, _net)
        
    def equivalence_recurrent(self, builder):
        _net = pybrainbridge._RecurrentNetwork()
        builder(_net)
        net = RecurrentNetwork()
        builder(net)
        super(TestNetworkEquivalence, self).equivalence_recurrent(net, _net)

    def testTwoLayerNetwork(self):
        self.equivalence_feed_forward(self.two_layer_network)

    def testSlicedNetwork(self):
        self.equivalence_feed_forward(self.sliced_connection_network)

    def testRecTwoLayerNetwork(self):
        self.equivalence_recurrent(self.rec_two_layer_network)

    def testRecThreeLayerNetwork(self):
        self.equivalence_recurrent(self.rec_three_layer_network)
        
    def testParametersDerivatives(self):
        rnet = pybrainbridge._RecurrentNetwork()
        self.lstm_network(rnet)
        rnet.sortModules()
        self.assert_(getattr(rnet, '_derivs', None) is not None)

        fnet = pybrainbridge._FeedForwardNetwork()
        self.two_layer_network(fnet)
        fnet.sortModules()
        self.assert_(getattr(fnet, '_derivs', None) is not None)
        
    def testTimesteps(self):
        _net = pybrainbridge._RecurrentNetwork()
        self.rec_two_layer_network(_net)
        _net.sortModules()
        
        netproxy = _net.proxies[_net]
        inproxy = _net.proxies[_net['in']]
        outproxy = _net.proxies[_net['out']]
        conproxy = _net.proxies[_net.connections[_net['in']][0]]
        rconproxy = _net.proxies[_net.recurrentConns[0]]
        
        proxies = netproxy, inproxy, outproxy, conproxy, rconproxy
        for proxy in proxies:
            self.assertEqual(proxy.get_mode(), 2)
            
        self.assertEqual(_net.offset, 0)
        for proxy in proxies:
            self.assertEqual(proxy.timestep(), 0,
                             "%s has wrong timestep." % proxy)

        _net.activate((0., 0.))
        for proxy in proxies:
            self.assertEqual(proxy.timestep(), 1)

        _net.activate((0., 0.))
        for proxy in proxies:
            self.assertEqual(proxy.timestep(), 2)

        _net.activate((0., 0.))
        for proxy in proxies:
            self.assertEqual(proxy.timestep(), 3)

        _net.backActivate((0., 0.))
        self.assertEqual(_net.offset, 2)
        for proxy in proxies:
            self.assertEqual(proxy.timestep(), 2)

        _net.backActivate((0., 0.))
        self.assertEqual(_net.offset, 1)
        for proxy in proxies:
            self.assertEqual(proxy.timestep(), 1)

        _net.backActivate((0., 0.))
        self.assertEqual(_net.offset, 0)
        for proxy in proxies:
            self.assertEqual(proxy.timestep(), 0)

    def testLstmNetwork(self):
        self.equivalence_recurrent(self.lstm_network)

    def testLstmCell(self):
        self.equivalence_recurrent(self.lstm_cell)

    def testWeirdNetwork(self):
        self.equivalence_feed_forward(self.weird_network)
        self.equivalence_recurrent(self.weird_network)
        
    def testCopyable(self):
        net = pybrainbridge._RecurrentNetwork()
        self.lstm_network(net)
        success = False
        try:
            copied = net.copy()
            success = True
        except TypeError, e:
            success = False
            self.assert_(success, e)
        
        
class TestNetworkUses(NetworkTestCase):
    
    def testBorderSwiping(self):
        size = 3
        dim = 3
        hsize = 1
        predefined = {}
        # assuming identical size in all dimensions
        dims = tuple([size]*dim)
        # also includes one dimension for the swipes
        hdims = tuple(list(dims)+[2**dim])
        inmod = LinearLayer(size**dim, name = 'input')
        inmesh = ModuleMesh.viewOnFlatLayer(inmod, dims, 'inmesh')
        outmod = LinearLayer(size**dim, name = 'output')
        outmesh = ModuleMesh.viewOnFlatLayer(outmod, dims, 'outmesh')
        hiddenmesh = ModuleMesh.constructWithLayers(TanhLayer, hsize, hdims, 'hidden')
        net = BorderSwipingNetwork(inmesh, hiddenmesh, outmesh, predefined = predefined)
        self.equivalence_feed_forward(net, net.convertToFastNetwork())
        
    def testMdlstm(self):
        net = FeedForwardNetwork()
        net.addInputModule(LinearLayer(1, name='in'))
        net.addModule(MDLSTMLayer(1, 1, name='hidden'))
        net.addOutputModule(LinearLayer(1, name='out'))
        net.addConnection(FullConnection(net['in'], net['hidden']))
        net.addConnection(FullConnection(net['hidden'], net['out']))
        net.sortModules()
        self.equivalence_feed_forward(net, net.convertToFastNetwork())
        

if __name__ == "__main__":
    unittest.main()  