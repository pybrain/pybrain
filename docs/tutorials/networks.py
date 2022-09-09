from __future__ import print_function

############################################################################
# PyBrain Tutorial "Networks, Modules, Connections"
#
# Author: Tom Schaul, tom@idsia.ch
############################################################################

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection, RecurrentNetwork
from pybrain.tools.shortcuts import buildNetwork

""" This tutorial will attempt to guide you for using one of PyBrain's most basic structural elements:
Networks, and with them Modules and Connections.

Let us start with a simple example, building a multi-layer-perceptron (MLP).

First we make a new network object: """

n = FeedForwardNetwork()

""" Next, we're constructing the input, hidden and output layers. """

inLayer = LinearLayer(2)
hiddenLayer = SigmoidLayer(3)
outLayer = LinearLayer(1)

""" (Note that we could also have used a hidden layer of type TanhLayer, LinearLayer, etc.)

Let's add them to the network: """

n.addInputModule(inLayer)
n.addModule(hiddenLayer)
n.addOutputModule(outLayer)

""" We still need to explicitly determine how they should be connected. For this we use the most
common connection type, which produces a full connectivity between two layers (or Modules, in general):
the 'FullConnection'. """

in2hidden = FullConnection(inLayer, hiddenLayer)
hidden2out = FullConnection(hiddenLayer, outLayer)
n.addConnection(in2hidden)
n.addConnection(hidden2out)

""" All the elements are in place now, so we can do the final step that makes our MLP usable,
which is to call the 'sortModules()' method. """

n.sortModules()

""" Let's see what we did. """

print(n)

""" One way of using the network is to call its 'activate()' method with an input to be transformed. """

print(n.activate([1, 2]))

""" We can access the trainable parameters (weights) of a connection directly, or read
all weights of the network at once. """

print(hidden2out.params)
print(n.params)

""" The former are the last slice of the latter. """

print(n.params[-3:] == hidden2out.params)

""" Ok, after having covered the basics, let's move on to some additional concepts.
First of all, we encourage you to name all modules, or connections you create, because that gives you
more readable printouts, and a very concise way of accessing them.

We now build an equivalent network to the one before, but with a more concise syntax:
"""
n2 = RecurrentNetwork(name='net2')
n2.addInputModule(LinearLayer(2, name='in'))
n2.addModule(SigmoidLayer(3, name='h'))
n2.addOutputModule(LinearLayer(1, name='out'))
n2.addConnection(FullConnection(n2['in'], n2['h'], name='c1'))
n2.addConnection(FullConnection(n2['h'], n2['out'], name='c2'))
n2.sortModules()

""" Printouts look more concise and readable: """
print(n2)

""" There is an even quicker way to build networks though, as long as their structure is nothing
more fancy than a stack of fully connected layers: """

n3 = buildNetwork(2, 3, 1, bias=False)

""" Recurrent networks are working in the same way, except that the recurrent connections
need to be explicitly declared upon construction.

We can modify our existing network 'net2' and add a recurrent connection on the hidden layer: """

n2.addRecurrentConnection(FullConnection(n2['h'], n2['h'], name='rec'))

""" After every structural modification, if we want ot use the network, we call 'sortModules()' again"""

n2.sortModules()
print(n2)

""" As the network is now recurrent, successive activations produce different outputs: """

print(n2.activate([1, 2]), end=' ')
print(n2.activate([1, 2]), end=' ')
print(n2.activate([1, 2]))

""" The 'reset()' method re-initializes the network, and with it sets the recurrent
activations to zero, so now we get the same results: """

n2.reset()
print(n2.activate([1, 2]), end=' ')
print(n2.activate([1, 2]), end=' ')
print(n2.activate([1, 2]))

""" This is already a good coverage of the basics, but if you're an advanced user
you might want to find out about the possibilities of nesting networks within
others, using weight-sharing, and more exotic types of networks, connections
and modules... but that goes beyond the scope of this tutorial.
"""

