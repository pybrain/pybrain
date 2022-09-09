from __future__ import print_function

#!/usr/bin/env python
""" Miniscule restricted Boltzmann machine usage example """

__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'

from pybrain.structure.networks.rbm import Rbm
from pybrain.unsupervised.trainers.rbm import (RbmGibbsTrainerConfig,
                                               RbmBernoulliTrainer)
from pybrain.datasets import UnsupervisedDataSet


ds = UnsupervisedDataSet(6)
ds.addSample([0, 1] * 3)
ds.addSample([1, 0] * 3)

cfg = RbmGibbsTrainerConfig()
cfg.maxIter = 3

rbm = Rbm.fromDims(6, 1)
trainer = RbmBernoulliTrainer(rbm, ds, cfg)
print(rbm.params, rbm.biasParams)
for _ in range(50):
    trainer.train()

print(rbm.params, rbm.biasParams)
