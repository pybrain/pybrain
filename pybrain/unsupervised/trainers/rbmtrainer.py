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
        if not (dataset.indim, dataset.outdim) == (rbm.indim, rbm.outdim):
            raise ValueError("Wrong dimension for dataset")   
        self._data = dataset
        
    data = property(_getData, _setData)
    
    def trainOnDataset(self, dataset, *args, **kwargs):
        self.data = dataset
        self._invRbm = Rbm(self.rbm.hiddendim, self.rbm.visibledim)
        self._invRbm.weights = self.rbm.weights.T
    
    def calcUpdateByRow(self, row):
        """Return a 3-tuple with update for weightmatrix, 
        hidden bias weights, visual bias weights."""
        # poshp: probabilities of hidden layer in positive phase
        poshp = self.rbm.activate(row)
        sampled = positive > numpy.random.rand(1, self.rbm.hiddendim)
        sampled = numpy.astype(sampled, numpy.int32)
        #
        recons = self._invRbm.activate(sampled)
        #
        # neghp: probs. of hidden layer in negative phase
        neghp = self.rbm.activate(recons)
        #
        # compute gradient update
        # pos: fraction from the positive phase
        pos = numpy.dot(row.T, poshp)
        # neg: fraction from the first cd-step
        neg = numpy.dot(recons.T, neghp)
        # gradient update for hidden bias 
        # the following lines are necessary if we 
        # have an input matrix, not a _row_
        #poshact = poshp.sum(axis=0)
        #neghact = neghp.sum(axis=0)
        # gradient update for visual bias
        #posvact = row.sum(axis=0)
        #negvact = recons.sum(axis=0)
        #
        return pos - neg, poshp - neghp, row - recons
