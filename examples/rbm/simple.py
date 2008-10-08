#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-

__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'
__version__ = '$Id$'


import scipy

from pybrain.structure.networks.rbm import Rbm
from pybrain.unsupervised.trainers.rbm import (RbmGibbsTrainerConfig, 
                                               RbmGibbsTrainer)
from pybrain.datasets import UnsupervisedDataSet


ds = UnsupervisedDataSet(6)
ds.addSample([0, 1] * 3)
ds.addSample([1, 0] * 3)

cfg = RbmGibbsTrainerConfig()
cfg.maxIter = 3

rbm = Rbm.fromDims(6, 1)
trainer = RbmGibbsTrainer(rbm, ds, cfg)
print rbm.weights, rbm.biasWeights
for _ in xrange(50):
    trainer.train()
    
print rbm.weights, rbm.biasWeights
