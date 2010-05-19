"""
    >>> from pybrain.tools.shortcuts     import buildNetwork
    >>> from pybrain.supervised.trainers import BackpropTrainer, RPropMinusTrainer
    >>> from pybrain.datasets import SupervisedDataSet, ImportanceDataSet
    >>> from scipy import random, array

Initialize random number generator

    >>> random.seed(42)

Create an XOR-dataset and a recurrent network

    >>> ds = ImportanceDataSet(2,2)
    >>> ds.addSample([0,0],[0, 1],  [1,0])
    >>> ds.addSample([0,1],[1, 10],  [1,0])
    >>> ds.addSample([1,0],[1, -1],  [1,0])
    >>> ds.addSample([1,1],[0, 0],  [1,0])
    >>> n = buildNetwork(ds.indim, 4, ds.outdim, recurrent=True)

Create and test backprop trainer

    >>> t = BackpropTrainer(n, learningrate = 0.01, momentum = 0.99, verbose = True)
    >>> t.trainOnDataset(ds, 4)
    Total error: 2.44696473875
    Total error: 1.97570498879
    Total error: 1.23940309483
    Total error: 0.546129967878
    >>> abs(n.params[10:15] - array([ -0.53868206, -0.54185834,  0.26726394, -1.90008234, -1.12114946])).round(5)
    array([ 0.,  0.,  0.,  0.,  0.])

Now the same for RPROP

    >>> t = RPropMinusTrainer(n, verbose = True)
    >>> t.trainOnDataset(ds, 4)
    epoch      0  total error      0.16818   avg weight       0.92638
    epoch      1  total error      0.15007   avg weight       0.92202
    epoch      2  total error      0.15572   avg weight       0.92684
    epoch      3  total error      0.13036   avg weight       0.92604
    >>> abs(n.params[5:10] - array([ -0.19241111,  1.43404022,  0.23062397, -0.40105413,  0.62100109])).round(5)
    array([ 0.,  0.,  0.,  0.,  0.])

"""

__author__ = 'Martin Felder, felder@in.tum.de'


from pybrain.tests import runModuleTestSuite

if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))

