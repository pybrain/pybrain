from pybrain.examples.datasets import AnBnCnDataSet
from pybrain.tools.shortcuts import buildSimpleNetwork
from pybrain.supervised import BackpropTrainer
from pybrain import FullConnection


def testTraining():
    d = AnBnCnDataSet()
    n = buildSimpleNetwork(0, 5, 1)
    n.addRecurrentConnection(FullConnection(n['h'], n['h']))
    t = BackpropTrainer(n, learningrate = 0.1, momentum = 0.0, verbose = True)
    t.trainOnDataset(d, 2000)
    #t._checkGradient(d)
    

if __name__ == '__main__':
    testTraining()