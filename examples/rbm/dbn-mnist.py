#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-

# Deep belief net applied to MNIST handwriting recognition dataset 

__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'
__version__ = '$Id$'


import scipy

from pybrain.datasets import UnsupervisedDataSet
from pybrain.unsupervised.trainers.deepbelief import DeepBeliefTrainer
from pybrain.tools.shortcuts import buildNetwork

from pybrainexamples.datasets.mnist import makeMnistDataSets


net = buildNetwork(784, 500, 500, 2000, bias=True)
train, test = makeMnistDataSets('/Users/bayerj/Desktop/MNIST/')

trainer = DeepBeliefTrainer(net, train)
trainer.train()

print "RBM Phase finished. Now backprop."
softmaxer = SoftmaxLayer(10)
con = FullConnection(net.outmodules[0], softmaxer)
net.addModule(softmaxer)
net.outmodules = [softmaxer]

trainer = BackpropTrainer(trainer, ds)
for i in xrange(sys.maxint):
    error = trainer.train()
    print "%i: %.2f" % (i, error)
