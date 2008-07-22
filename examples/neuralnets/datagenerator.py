import pylab as p
import numpy as np
from pybrain.datasets import ClassificationDataSet

def generateClassificationData(size):
    """ generate a set of points in 2D belonging to three different classes """
    means = [(-1,0),(2,4),(3,1)]
    cov = [p.diag([1,1]), p.diag([0.5,1.2]), p.diag([1.5,0.7])]
    targets = []
    inputs = []
    for c in range(3):
        xy = np.random.multivariate_normal(means[c],cov[c],size=size)
        inputs.extend(xy)
        targets.extend([c]*size)
    dataset = ClassificationDataSet(2, 1, nb_classes=3)
    dataset.setField('input',  p.array(inputs))
    dataset.setField('target', p.array(targets).reshape(size*3,1))
    return dataset

def generateGridData(min, max, step):
    """ generates a dataset containing a regular square grid of points """
    x = np.arange(min, max, step)
    y = np.arange(min, max, step)
    X, Y = np.meshgrid(x, y)
    shape = X.shape
    # need column vectors in dataset, not arrays
    ds = ClassificationDataSet(2,1)
    ds.setField('input',  np.concatenate((X.reshape(X.size, 1),Y.reshape(X.size, 1)), 1))
    ds.setField('target', np.zeros([X.size,1]))
    ds._convertToOneOfMany()
    return (ds, X, Y)
    

def plotData(ds):
    p.hold(True)
    for c in range(3):
        here, dumm = np.where(ds['class']==c)
        p.plot(ds['input'][here,0],ds['input'][here,1],'o')
    
if __name__ == '__main__':    
    plotData(generateClassificationData(150))
    p.show()
