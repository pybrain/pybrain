#! /usr/bin/env python2.5
# -*- coding: utf-8 -*

"""

    >>> from scipy import array, ones, zeros
    >>> from ctypes import pointer


Construction of a layer structure
----------------------------------

    >>> inputbuffer = array((0,))
    >>> outputbuffer = array((0,))
    >>> inerror = array((0,))
    >>> outerror = array((0,))
    >>> l = c_layer(1, 1, inputbuffer, outputbuffer, inerror, outerror)
    
    
Construction of a connection structure
--------------------------------------

    >>> con = c_connection()
    >>> fc = c_full_connection()
    >>> a, b = array((0., 1., 2.)), array((2., 3., 4.))
    >>> weights = c_parameter_container(a, b)
    >>> fc.weights = weights
    >>> con.type = 1
    >>> con.internal.full_connection_p = pointer(fc)


Adding connections to a layer
-----------------------------

    >>> l.add_outgoing_connection(con)
    >>> l.outgoing_n
    1
    >>> l.outgoing_p[0].internal.full_connection_p.contents.weights.contents_p[0:3]
    [0.0, 1.0, 2.0]
    >>> l.add_outgoing_connection(con)
    >>> l.outgoing_n
    2
    >>> l.outgoing_p[0].internal.full_connection_p.contents.weights.contents_p[0:3]
    [0.0, 1.0, 2.0]
    >>> l.outgoing_p[1].internal.full_connection_p.contents.weights.contents_p[0:3]
    [0.0, 1.0, 2.0]


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
    array([  1.25, -20.4 ])


Growing of buffers and making sure that nothing is lost
-------------------------------------------------------

    >>> net = _FeedForwardNetwork()
    >>> in_ = LinearLayer(1, 'in')
    >>> out = LinearLayer(1, 'out')
    >>> net.addInputModule(in_)
    >>> net.addOutputModule(out)
    >>> net.sortModules()

    >>> net['in'].inputbuffer[:] = array((0.5))
    >>> net['in'].outputbuffer[:] = array((1.5))
    >>> net['in'].inputerror[:] = array((0.2))
    >>> net['in'].outputerror[:] = array((1.2))

    >>> net['out'].inputbuffer[:] = array((0.3))
    >>> net['out'].outputbuffer[:] = array((1.3))
    >>> net['out'].inputerror[:] = array((0.4))
    >>> net['out'].outputerror[:] = array((1.4))

    >>> net._growBuffers()
    >>> net.sortModules()
    
    >>> net['in'].inputbuffer
    array([[ 0.5],
           [ 0. ]])

    >>> net['in'].outputbuffer
    array([[ 1.5],
           [ 0. ]])

    >>> net['in'].inputerror
    array([[ 0.2],
           [ 0. ]])

    >>> net['in'].outputerror
    array([[ 1.2],
           [ 0. ]])

    >>> net['out'].inputbuffer
    array([[ 0.3],
           [ 0. ]])

    >>> net['out'].outputbuffer
    array([[ 1.3],
           [ 0. ]])

    >>> net['out'].inputerror
    array([[ 0.4],
           [ 0. ]])

    >>> net['out'].outputerror
    array([[ 1.4],
           [ 0. ]])

   >>> net._growBuffers()

   >>> net['in'].inputbuffer
   array([[ 0.5],
          [ 0. ],
          [ 0. ],
          [ 0. ]])

   >>> net['in'].outputbuffer
   array([[ 1.5],
          [ 0. ],
          [ 0. ],
          [ 0. ]])

   >>> net['in'].inputerror
   array([[ 0.2],
          [ 0. ],
          [ 0. ],
          [ 0. ]])

   >>> net['in'].outputerror
   array([[ 1.2],
          [ 0. ],
          [ 0. ],
          [ 0. ]])

   >>> net['out'].inputbuffer
   array([[ 0.3],
          [ 0. ],
          [ 0. ],
          [ 0. ]])

   >>> net['out'].outputbuffer
   array([[ 1.3],
          [ 0. ],
          [ 0. ],
          [ 0. ]])

   >>> net['out'].inputerror
   array([[ 0.4],
          [ 0. ],
          [ 0. ],
          [ 0. ]])

   >>> net['out'].outputerror
   array([[ 1.4],
          [ 0. ],
          [ 0. ],
          [ 0. ]])


Construction of a Network containing a single LstmLayer
-------------------------------------------------------

    >>> net = _RecurrentNetwork()
    >>> l = LSTMLayer(1)
    >>> net.addRecurrentConnection(FullConnection(l, l))
    >>> net.addInputModule(l)
    >>> net.outmodules.append(l)
    >>> net.sortModules()
    
    
Construction of a Network with a TanhLayer
------------------------------------------

    >>> from pybrain.structure import TanhLayer
    >>> net = _FeedForwardNetwork()
    >>> l = TanhLayer(2)
    >>> net.addInputModule(l)
    >>> net.outmodules.append(l)
    >>> net.sortModules()
    >>> net.activate((2, 3))
    array([ 0.96402758,  0.99505475])


Construction of a Network with a SoftmaxLayer
------------------------------------------

    >>> from pybrain.structure import SoftmaxLayer
    >>> net = _FeedForwardNetwork()
    >>> l = SoftmaxLayer(2)
    >>> net.addInputModule(l)
    >>> net.outmodules.append(l)
    >>> net.sortModules()
    >>> net.activate((2, 3))
    array([ 0.26894142,  0.73105858])


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
