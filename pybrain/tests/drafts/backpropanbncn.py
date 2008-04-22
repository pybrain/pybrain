from pybrain.examples.datasets import AnBnCnDataSet
from pybrain.supervised import BackpropTrainer
from pybrain import FullConnection, buildNetwork


def testTraining():
    d = AnBnCnDataSet()
    n = buildNetwork(0, 5, 1, bias = False)
    n.addRecurrentConnection(FullConnection(n['h'], n['h']))
    t = BackpropTrainer(n, learningrate = 0.1, momentum = 0.0, verbose = True)
    t.trainOnDataset(d, 2000)
    #t._checkGradient(d)
    

if __name__ == '__main__':
    testTraining()