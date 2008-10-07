#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-

__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'
__version__ = '$Id$'


from pybrain.structure import (LinearLayer, SigmoidLayer, FullConnection, 
                               BiasUnit, FeedForwardNetwork)


def buildRbm(visibledim, hiddendim, weights=None, biasweights=None):
    """Return a restricted Boltzmann machine of the given dimensions with the
    given distributions."""
    net = FeedForwardNetwork()
    bias = BiasUnit('bias')
    visible = LinearLayer(visibledim, 'visible')
    hidden = SigmoidLayer(hiddendim, 'hidden')
    con1 = FullConnection(visible, hidden)
    con2 = FullConnection(bias, hidden)
    if weights is not None:
        con1.params[:] = weights
    if biasweights is not None:
        con2.params[:] = biasweights

    net.addInputModule(visible)
    net.addModule(bias)
    net.addOutputModule(hidden)
    net.addConnection(con1)
    net.addConnection(con2)
    net.sortModules()
    return net
    

def invRbm(rbm):
    """Return the inverse rbm of an rbm."""
    visibledim = rbm.outdim
    hiddendim = rbm.indim
    # TODO: check if shape is correct
    params = rbm.connections[rbm['visible']][0].params[:visibledim * hiddendim]
    params = params.reshape(visibledim, hiddendim).T.flatten()
    return buildRbm(visibledim, hiddendim, weights=params)