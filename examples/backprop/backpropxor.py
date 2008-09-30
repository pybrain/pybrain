__author__ = 'Tom Schaul, tom@idsia.ch'

from datasets import SequentialXORDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer


def testTraining():
    d = SequentialXORDataSet()
    n = buildNetwork(d.indim, 4, d.outdim, recurrent=True)
    t = BackpropTrainer(n, learningrate = 0.01, momentum = 0.99, verbose = True)
    t.trainOnDataset(d, 1000)
    t.testOnData(verbose= True)
    
    
if __name__ == '__main__':
    testTraining()