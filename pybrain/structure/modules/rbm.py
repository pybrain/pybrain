#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-


__author__ = ('Christian Osendorfer, osendorf@in.tum.de;'
              'Justin S Bayer, bayerj@in.tum.de')


from pybrain.structure import Network

class Rbm(Network):

    def __init__(self, visibledim, hiddendim):
        super(Rbm, self).__init__()
        self.visibledim = visibledim
        self.hiddendim = hiddendim
        bias = BiasUnit()
        ll = LinearLayer(visibledim))
        sl = SigmoidLayer(hiddendim))
        self.addInputModule(ll)
        self.addOutputModule(sl)
        self.addConnection(FullConnection(ll, sl))
        self.addConnection(FullConnection(bias, sl))

    def _getWeights(self):
        return self.connections[0].params

    def _setWeights(self, val):
        self.connections[0].params[:] = val
    
    weights = property(_getWeights, _setWeights)

