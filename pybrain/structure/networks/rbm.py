#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-


__author__ = ('Christian Osendorfer, osendorf@in.tum.de;'
              'Justin S Bayer, bayerj@in.tum.de')


from pybrain.structure.modules import BiasUnit, SigmoidLayer, LinearLayer
from pybrain.structure.connections import FullConnection
from pybrain.structure.networks import Network

class Rbm(Network):

    def __init__(self, visibledim, hiddendim):
        super(Rbm, self).__init__()
        self.visibledim = visibledim
        self.hiddendim = hiddendim
        bias = BiasUnit()
        
        self.visibleLayer = LinearLayer(visibledim)
        self.hidddenLayer = SigmoidLayer(hiddendim)
        self.fullConnection = FullConnection(self.visibleLayer, self.hidddenLayer, self.connNames.next())
        self.biasConnection = FullConnection(bias, self.hidddenLayer)
        
        self.addInputModule(self.visibleLayer)
        self.addOutputModule(self.hiddenLayer)
        self.addModule(bias)
        self.addConnection(self.fullConnection)
        self.addConnection(self.biasConnection)
        self.sortModules()

    def _getWeights(self):
        return self.connections[0].params

    def _setWeights(self, val):
        self.connections[0].params[:] = val
    
    weights = property(_getWeights, _setWeights)
    
    def _getBiasWeights(self):
        return self.connections[1].params
        
    def _setBiasWeights(self, value):
        self.connections[1].paramse = value
        
    biasWeights = property(_getBiasWeights, _setBiasWeights)

