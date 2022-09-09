#!/usr/bin/env python
# generates some simple example data sets
__author__ = "Martin Felder"
__version__ = '$Id$'

import numpy as np
from numpy.random import multivariate_normal, rand
from scipy import diag
from pylab import show, hold, plot

from pybrain.datasets import ClassificationDataSet, SequenceClassificationDataSet

def generateClassificationData(size, nClasses=3):
    """ generate a set of points in 2D belonging to two or three different classes """
    if nClasses==3:
        means = [(-1,0),(2,4),(3,1)]
    else:
        means = [(-2,0),(2,1),(6,0)]

    cov = [diag([1,1]), diag([0.5,1.2]), diag([1.5,0.7])]
    dataset = ClassificationDataSet(2, 1, nb_classes=nClasses)
    for _ in range(size):
        for c in range(3):
            input = multivariate_normal(means[c],cov[c])
            dataset.addSample(input, [c%nClasses])
    dataset.assignClasses()
    return dataset


def generateGridData(x,y, return_ticks=False):
    """ Generates a dataset containing a regular grid of points. The x and y arguments
    contain start, end, and step each. Returns the dataset and the x and y mesh or ticks."""
    x = np.arange(x[0], x[1], x[2])
    y = np.arange(y[0], y[1], y[2])
    X, Y = np.meshgrid(x, y)
    shape = X.shape
    # need column vectors in dataset, not arrays
    ds = ClassificationDataSet(2,1)
    ds.setField('input',  np.concatenate((X.reshape(X.size, 1),Y.reshape(X.size, 1)), 1))
    ds.setField('target', np.zeros([X.size,1]))
    ds._convertToOneOfMany()
    if return_ticks:
        return (ds, x, y)
    else:
        return (ds, X, Y)


def generateNoisySines( npoints, nseq, noise=0.3 ):
    """ construct a 2-class dataset out of noisy sines """
    x = np.arange(npoints)/float(npoints) * 20.
    y1 = np.sin(x+rand(1)*3.)
    y2 = np.sin(x/2.+rand(1)*3.)
    DS = SequenceClassificationDataSet(1,1, nb_classes=2)
    for _ in range(nseq):
        DS.newSequence()
        buf = rand(npoints)*noise + y1 + (rand(1)-0.5)*noise
        for i in range(npoints):
            DS.addSample([buf[i]],[0])
        DS.newSequence()
        buf = rand(npoints)*noise + y2 + (rand(1)-0.5)*noise
        for i in range(npoints):
            DS.addSample([buf[i]],[1])
    return DS

def plotData(ds):
    hold(True)
    for c in range(ds.nClasses):
        here, _ = np.where(ds['class']==c)
        plot(ds['input'][here,0],ds['input'][here,1],'o')


if __name__ == '__main__':
    plotData(generateClassificationData(150))
    show()
