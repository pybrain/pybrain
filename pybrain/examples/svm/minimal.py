#!/usr/bin/python

__author__ = 'Michael Isik'


from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers.svm import SVMTrainer
from pybrain.structure.modules.svm import KT,SVM
#from pybrain.examples.svm.debug import dbg,tracedm
from numpy import array



if __name__ == "__main__":

    dataset = SupervisedDataSet(2,1)
    dataset.addSample( [ 0.10, 0.50 ], [2] )
    dataset.addSample( [ 1.20, 1.60 ], [2] )
    dataset.addSample( [ 0.20, 0.80 ], [2] )
    dataset.addSample( [ 0.30, 1.70 ], [8] )
    dataset.addSample( [ 1.40, 0.10 ], [8] )

    module  = SVM( dataset.indim, dataset.outdim, KT.RBF )
    trainer = SVMTrainer(module, cost=20.)
    trainer.setData( dataset )


    trainer.train()

    wrongcount = 0
    for i in range(dataset.getLength()):
        xi, yi = dataset.getSample(i)
        out = array([[0]])
        module._forwardImplementation( array([xi]), out )
        print " ", i,":   x =", xi, " out =", out[0], " target =", yi[0], "  f = %.2f"%module.rawOutput(xi) 
        if out[0] != yi: wrongcount += 1

    print "iterations =", trainer.stepno
    print "samples    =", dataset.getLength()
    print "false_num  =", wrongcount



