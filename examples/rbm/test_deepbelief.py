#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""

    >>> from pybrain.tools.shortcuts import buildNetwork
    >>> from pybrain.unsupervised.trainers.deepbelief import DeepBeliefTrainer
    >>> from pybrain.datasets import UnsupervisedDataSet
    >>> net = buildNetwork(2, 2, 2, 2, bias=True)
    >>> ds = UnsupervisedDataSet(2)
    >>> trainer = DeepBeliefTrainer(net, ds)
    >>> for rbm in trainer.iterRbms():
    ...   print rbm



"""


__author__ = 'Justin Bayer, bayerj@in.tum.de'
__version__ = '$Id$'


from pybrain.tests import runModuleTestSuite


if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))