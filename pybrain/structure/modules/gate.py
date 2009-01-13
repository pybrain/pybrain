#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'
__version__ = '$Id$'


from pybrain.structure.modules.module import Module
from pybrain.structure.modules.neuronlayer import NeuronLayer
from pybrain.tools.functions import sigmoid, sigmoidPrime


class GateLayer(NeuronLayer):
    """Layer that implements pairwise input multiplication, with one element of
    the pair being squashed.
    
    If a GateLayer of size n is created, it will have 2 * n inputs and n 
    outputs. The i'th output is calculated as sigmoid(I_i) * I_(i + n) where I
    is the vector of inputs."""
    
    def __init__(self, dim, name=None):
        Module.__init__(self, 2 * dim, dim, name)
    
    def _forwardImplementation(self, inbuf, outbuf):
        outbuf += sigmoid(inbuf[:self.outdim]) * inbuf[self.outdim:]
        
    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:self.outdim] += (sigmoidPrime(inbuf[:self.outdim])  
                                * inbuf[self.outdim:]
                                * outerr)
        inerr[self.outdim:] += (sigmoid(inbuf[:self.outdim]) 
                                * outerr)