#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-

__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'
__version__ = '$Id$'


from pybrain.structure.networks.rbm import buildRbm
from pybrain.unsupervised.trainers.rbmtrainer import RbmGibbsTrainerConfig, RbmGibbsTrainer
from pybrain.datasets import SupervisedDataSet


ds = SupervisedDataSet(6, 2)
ds.addSample([0, 1] * 3, [0])
ds.addSample([1, 0] * 3, [0])

cfg = RbmGibbsTrainerConfig()
cfg.maxIter = 3

rbm = buildRbm(6, 1)
trainer = RbmGibbsTrainer(rbm, ds, cfg)
print rbm.params
trainer.train()
print rbm.params
