#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-

# Miniscule deep belief net example 

__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'
__version__ = '$Id$'


from pybrain.datasets import UnsupervisedDataSet
from pybrain.unsupervised.trainers.deepbelief import DeepBeliefTrainer
from pybrain.tools.shortcuts import buildNetwork


ds = UnsupervisedDataSet(6)
ds.addSample([0, 1] * 3)
ds.addSample([1, 0] * 3)

net = buildNetwork(6, 2, 2, 2, bias=True)
params = net.params.copy()

trainer = DeepBeliefTrainer(net, ds)

trainer.train()

print params == net.params