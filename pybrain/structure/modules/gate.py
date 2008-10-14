#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'
__version__ = '$Id$'


from pybrain.structure.modules.neuronlayer import NeuronLayer


class GateLayer(NeuronLayer):
    """Layer that implements pairwise input multiplication.
    
    If a GateLayer of size n is created, it will have 2 * n inputs and n 
    outputs. The i'th output is calculated as I_i * I_(i + n) where I is the 
    vector of inputs."""
    
    def __init__(self, dim, name=None):
        Module.__init__(self, 2 * dim, dim, name)
    
    def _forwardImplementation(self, inbuf, outbuf):
        outbuf += inbuf[:self.dim / 2] * inbuf[self.dim / 2:]
        
    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:self.dim / 2] += inbuf[self.dim / 2:] * outerr[:self.dim / 2]
        inerr[self.dim / 2:] += inbuf[:self.dim / 2] * outerr[self.dim / 2:]