#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-

__author__ = ('Christian Osendorfer, osendorf@in.tum.de;'
              'Justin S Bayer, bayerj@in.tum.de')


from scipy import array, random, outer

from pybrain.structure.networks.rbm import Rbm
from pybrain.supervised.trainers import Trainer


class RbmTrainer(Trainer):

    def __init__(self, rbm):
        self.rbm = rbm
        self._invRbm = Rbm(self.rbm.hiddendim, self.rbm.visibledim)
        self._invRbm.weights = self.rbm.weights.T
    
    def _getData(self): 
        return self._data
        
    def _setData(self, dataset):
        if not dataset.indim == self.rbm.visible:
            raise ValueError("Wrong dimension for dataset")   
        self._data = dataset
        
    data = property(_getData, _setData)
    
    def trainOnDataset(self, dataset, learningrate=.1, batchsize=50):
        for rows in dataset.randomBatches('sample', batchsize):
            uWeights, uHidBias, uVisBias = self.calcUpdateByRows(rows)
            # Normalization and learning rate factor
            a = learningrate / len(rows)
            self.rbm.weights += a * uWeights
            self._invRbm.biasWeights += a * uHidBias
            self.rbm.biasWeights += a * uVisBias
    
    def calcUpdateByRow(self, row):
        """Return a 3-tuple consiting of updates for (weightmatrix, 
        hidden bias weights, visible bias weights)."""
        # Probabilities of hidden layer in positive phase
        poshp = self.rbm.activate(row)
        # Stochastic Binary Units: Bernoulli sampling
        sampled = poshp > random.rand(1, self.rbm.hiddendim)
        sampled = sampled.astype('int32')
        # "Reconstruction" of the input
        recons = self._invRbm.activate(sampled)
        # Probabilities of hidden layer in negative phase
        neghp = self.rbm.activate(recons)
        # Gradient update for weights
        pos = outer(row, poshp)       # Fraction from the positive phase
        neg = outer(recons, neghp)    # Fraction from the first CD step
        return pos - neg, poshp - neghp, row - recons
        
    def calcUpdateByRows(self, rows):
        """Return a 3-tuple constisting of update for (weightmatrix, 
        hidden bias weights, visible bias weights)."""
        # Probabilities of hidden layer in positive phase
        poshp = array([self.rbm.activate(row) for row in rows])
        # Stochastic Binary Units: Bernoulli sampling
        sample_shape = rows.shape[0], self.rbm.hiddendim
        sampled = poshp > random.random(sample_shape)
        sampled = sampled.astype('int32')
        # "Reconstruction" of the input
        recons = array([self._invRbm.activate(sample) for sample in sampled])
        # Probabilities of hidden layer in negative phase
        neghp = array([self.rbm.activate(row) for row in recons])
        # Gradient update for weights
        pos = outer(row, poshp.T)    # Fraction from the positive phase
        neg = outer(recons, neghp.T) # Fraction from the first CD step
        # Gradient update for hidden bias 
        poshact = poshp.sum(axis=0)
        neghact = neghp.sum(axis=0)
        # Gradient update for visual bias
        posvact = rows.sum(axis=0)
        negvact = recons.sum(axis=0)

        try:
            weightupdate = pos - neg
        except: 
            print pos.shape, neg.shape, 0
            return 0, 0, 0
        hiddenbiasupdate = posvact - negvact 
        visiblebiasupdate = poshact - neghact
        return weightupdate, hiddenbiasupdate, visiblebiasupdate