#! /usr/bin/env python2.5
# -*- coding: utf-8 -*

"""

    >>> from scipy import array


Construction of a layer structure
----------------------------------

    >>> inputbuffer = array((0,))
    >>> outputbuffer = array((0,))
    >>> inerror = array((0,))
    >>> outerror = array((0,))
    >>> m = c_layer(1, 1, inputbuffer, outputbuffer, inerror, outerror)
    
    
Construction of a connection structure
--------------------------------------

    TODO

Construction of a Network from Pybrain structures
-------------------------------------------------    
    
    >>> net = _FeedForwardNetwork()
    >>> in_ = LinearLayer(3)
    >>> out = LinearLayer(2)
    >>> con = FullConnection(in_, out)
    >>> con._params = array((2.5, 3.0, 1.0, 3.0, 4.0, -3.0))
    >>> net.addInputModule(in_)
    >>> net.addOutputModule(out)
    >>> net.addConnection(con)
    >>> net.sortModules()
    >>> net.modulearray is not None
    True

    >>> net.activate(array((1.2, -2.25, 5.0)))
    [array([ 20.75, -18.15])]
    
    
Construction of a Network containing a single LstmLayer
-------------------------------------------------------

    >>> net = _RecurrentNetwork()
    >>> l = LSTMLayer(1)
    >>> net.addRecurrentConnection(FullConnection(l, l))
    >>> net.addInputModule(l)
    >>> net.outmodules.append(l)
    >>> net.sortModules()
    
    
Using a backprop trainer on a CNetwork
--------------------------------------

    >>> net = _FeedForwardNetwork()
    >>> inpt = LinearLayer(2)
    >>> hidden = SigmoidLayer(3)
    >>> outpt = SigmoidLayer(1)
    
    >>> con1 = FullConnection(inpt, hidden)
    >>> con2 = FullConnection(hidden, outpt)
    
    >>> net.addInputModule(inpt)
    >>> net.addModule(hidden)
    >>> net.addOutputModule(outpt)
    >>> net.addConnection(con1)
    >>> net.addConnection(con2)
    >>> net.sortModules()
    
    >>> import pybrain.datasets
    >>> ds = pybrain.datasets.SupervisedDataSet(2, 1)
    >>> ds.addSample(array([0, 0]), array([0]))
    >>> ds.addSample(array([0, 1]), array([1]))
    >>> ds.addSample(array([1, 0]), array([1]))
    >>> ds.addSample(array([1, 1]), array([0]))

    >>> from pybrain.supervised.trainers import BackpropTrainer
    
    >>> trainer = BackpropTrainer(net, dataset=ds)
    >>> trainer.trainEpochs(10)
    
"""


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


from arac.pybrainbridge import _Network, _FeedForwardNetwork, _RecurrentNetwork
from arac.structure import c_parameter_container, c_bias_layer, \
    c_identity_layer, c_sigmoid_layer, c_lstm_layer, c_identity_connection, \
    c_full_connection, c_layer, c_connection
from pybrain.structure import LinearLayer, SigmoidLayer, LSTMLayer, \
    IdentityConnection, FullConnection


if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
