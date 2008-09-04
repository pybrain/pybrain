import numpy as np
from pybrain.datasets import ClassificationDataSet
from numpy.random import multivariate_normal
from scipy import diag
from pylab import show, hold, plot #@UnresolvedImport

def generateClassificationData(size):
    """ generate a set of points in 2D belonging to three different classes """
    means = [(-1,0),(2,4),(3,1)]
    cov = [diag([1,1]), diag([0.5,1.2]), diag([1.5,0.7])]
    dataset = ClassificationDataSet(2, 1, nb_classes=3)
    for _ in xrange(size):
        for c in range(3):
            input = multivariate_normal(means[c],cov[c])
            dataset.addSample(input, [c])
    return dataset

def generateGridData(min, max, step):
    """ generates a dataset containing a regular square grid of points """
    x = np.arange(min, max, step)
    y = np.arange(min, max, step)
    X, Y = np.meshgrid(x, y)
    # need column vectors in dataset, not arrays
    ds = ClassificationDataSet(2,1)
    ds.setField('input',  np.concatenate((X.reshape(X.size, 1),Y.reshape(X.size, 1)), 1))
    ds.setField('target', np.zeros([X.size,1]))
    ds._convertToOneOfMany()
    return (ds, X, Y)
    

def plotData(ds):
    hold(True)
    for c in range(3):
        here, _ = np.where(ds['class']==c)
        plot(ds['input'][here,0],ds['input'][here,1],'o')
    
if __name__ == '__main__':    
    plotData(generateClassificationData(150))
    show()
