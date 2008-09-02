#!/usr/bin/python

__author__ = 'Michael Isik'


from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers.mcsvm import MCSVMTrainer
from pybrain.structure.modules.svm     import KT
from pybrain.structure.modules.mcsvm   import MCSVMOneAgainstAll


if __name__ == "__main__":

    dataset = SupervisedDataSet(2,1)
    dataset.addSample( [ 0.1, 0.5 ]   , [1] )
    dataset.addSample( [ 1.2, 1.6 ]   , [2] )
    dataset.addSample( [ 0.20, 0.80 ] , [3] )

#    module  = MCSVMOneAgainstOne( dataset.indim, dataset.outdim, KT.RBF )
#    module  = MCSVM( dataset.indim, dataset.outdim, KT.RBF )
#    trainer = MCSVMTrainer(module, cost={1:4., 2:0.5, 3:7})

    module  = MCSVMOneAgainstAll( dataset.indim, dataset.outdim, KT.RBF )
    trainer = MCSVMTrainer(module, cost=7)

    trainer.setData( dataset )

    trainer.train()


    wrongcount = 0
    for i in range(dataset.getLength()):
        xi, yi = dataset.getSample(i)
        out = module.classify(xi)
        print " ", i,":   x =", xi, " out =", out, " target =", yi[0]
        if out != yi: wrongcount += 1

    print "samples    =", dataset.getLength()
    print "false_num  =", wrongcount



