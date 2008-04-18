#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-


__author__ = ('Christian Osendorfer, osendorf@in.tum.de;'
              'Justin S Bayer, bayerj@in.tum.de')


from pybrain.supervised.trainers import Trainer
import numpy


class RbmTrainer(Trainer):

    def __init__(self, rbm):
        self.rbm = rbm
        self._data = None    
    
    def _getData(self): 
        return self._data
        
    def _setData(self, dataset):
        if not dataset.indim == rbm.visible:
            raise ValueError("Wrong dimension for dataset")   
        self._data = dataset
        
    data = property(_getData, _setData)
    
    def trainOnDataset(self, dataset, learningrate=.1, batchsize=50):
        self._invRbm = Rbm(self.rbm.hiddendim, self.rbm.visibledim)
        self._invRbm.weights = self.rbm.weights.T
        for rows in dataset.randomBatches('sample', batchsize):
            uWeights, uHidBias, uVisBias = self.calcUpdateByRows(rows)
            # Normalization and learning rate factor
            a = learningrate / len(rows)
            self.rbm.weights += a * uWeights
            self._invRbm.biasWeights += a * uHidBias
            self.rbm.biasWeights += a * uVisBias
    
    def calcUpdateByRow(self, row):
        """Return a 3-tuple consiting of updates for (weightmatrix, 
        hidden bias weights, visual bias weights)."""
        # Probabilities of hidden layer in positive phase
        poshp = self.rbm.activate(row)
        # Stochastic Binary Units: Bernoulli sampling
        sampled = positive > numpy.random.rand(1, self.rbm.hiddendim)
        sampled = numpy.astype(sampled, numpy.int32)
        # "Reconstruction" of the input
        recons = self._invRbm.activate(sampled)
        # Probabilities of hidden layer in negative phase
        neghp = self.rbm.activate(recons)
        # Gradient update for weights
        pos = numpy.dot(row.T, poshp)       # Fraction from the positive phase
        neg = numpy.dot(recons.T, neghp)    # Fraction from the first CD step
        return pos - neg, poshp - neghp, row - recons
        
    def calcUpdateByRows(self, rows):
        """Return a 3-tuple constisting of update for (weightmatrix, 
        hidden bias weights, visual bias weights)."""
        # Probabilities of hidden layer in positive phase
        poshp = array([self.rbm.activate(row) for row in rows])
        # Stochastic Binary Units: Bernoulli sampling
        sample_shape = self.rows.shape[0], self.rbm.hiddendim
        sampled = positive > numpy.random.random(shape)
        sampled.dtype = 'int32'
        # "Reconstruction" of the input
        recons = array([self._invRbm.activate(sample) for sample in sampled])
        # Probabilities of hidden layer in negative phase
        neghp = array([self.rbm.activate(row) for row in recons])
        # Gradient update for weights
        pos = numpy.dot(row.T, poshp)    # Fraction from the positive phase
        neg = numpy.dot(recons.T, neghp) # Fraction from the first CD step
        # Gradient update for hidden bias 
        poshact = poshp.sum(axis=0)
        neghact = neghp.sum(axis=0)
        # Gradient update for visual bias
        posvact = rows.sum(axis=0)
        negvact = recons.sum(axis=0)
        return pos - neg, posvact - negvact, poshact - neghact