#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-


__author__ = ('Christian Osendorfer, osendorf@in.tum.de;'
              'Justin S Bayer, bayerj@in.tum.de')


from pybrain.structure.modules import BiasUnit, SigmoidLayer, LinearLayer
from pybrain.structure.connections import FullConnection
from pybrain.structure.networks import Network
from pybrain.structure.modules.module import Module


class Rbm(Network):

    def __init__(self, visibledim, hiddendim):
        super(Rbm, self).__init__()
        Module.__init__(self, visibledim, hiddendim)
        self.visibledim = visibledim
        self.hiddendim = hiddendim
        
        bias = BiasUnit()
        vl = self.visibleLayer = LinearLayer(visibledim)
        hl = self.hidddenLayer = SigmoidLayer(hiddendim)
        fc = self.fullConnection = FullConnection(vl, hl)
        bc = self.biasConnection = FullConnection(bias, hl)
        
        self.addInputModule(vl)
        self.addOutputModule(hl)
        self.addConnection(fc)
        self.addConnection(bc)
        
        self.sortModules()

    def _getWeights(self):
        return self.fullConnection.params

    def _setWeights(self, val):
        self.fullConnection.params[:] = val
    
    weights = property(_getWeights, _setWeights)
    
    def _getBiasWeights(self):
        return self.biasConnection.params
        
    def _setBiasWeights(self, value):
        self.biasConnection.params = value
        
    biasWeights = property(_getBiasWeights, _setBiasWeights)

