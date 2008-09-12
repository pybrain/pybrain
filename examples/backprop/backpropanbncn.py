__author__ = 'Tom Schaul, tom@idsia.ch and Daan Wierstra'

from datasets import AnBnCnDataSet
from pybrain.supervised import BackpropTrainer
from pybrain.structure import FullConnection, RecurrentNetwork, TanhLayer, LinearLayer, BiasUnit


def testTraining():
    d = AnBnCnDataSet()
    hsize = 2
    n = RecurrentNetwork()
    n.addModule(TanhLayer(hsize, name = 'h'))
    n.addModule(BiasUnit(name = 'bias'))
    n.addOutputModule(LinearLayer(1, name = 'out'))
    n.addConnection(FullConnection(n['bias'], n['h']))
    n.addConnection(FullConnection(n['h'], n['out']))
    n.addRecurrentConnection(FullConnection(n['h'], n['h']))
    n.sortModules()
    assert n.indim == 0
    assert n.outdim == 1
    assert n.paramdim == hsize*(hsize+2)
    t = BackpropTrainer(n, learningrate = 0.1, momentum = 0.0, verbose = True)
    t.trainOnDataset(d, 200)
    print 'Final weights:', n.params

if __name__ == '__main__':
    testTraining()