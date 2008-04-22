__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.examples.datasets import XORDataSet, SequentialXORDataSet
from pybrain import buildNetwork
from pybrain.supervised import BackpropTrainer


def testTraining():
    #d = XORDataSet()
    d = SequentialXORDataSet()
    n = buildNetwork(d.indim, 4, d.outdim)
    t = BackpropTrainer(n, learningrate = 0.01, momentum = 0.99, verbose = True)
    t.trainOnDataset(d, 1000)
    t.testOnData(verbose= True)
    
    
if __name__ == '__main__':
    testTraining()