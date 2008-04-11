#!/usr/bin/python

__author__ = 'Michael Isik'


from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers.svm import SVMTrainer
from pybrain.structure.modules.svm import SVM, SimplePolyKernel


from numpy import array



if __name__ == "__main__":

    dataset = SupervisedDataSet(2,1)
    dataset.addSample( [ 0.1  , 0.5  ], [23] )
    dataset.addSample( [ 1.2  , 1.6  ], [23] )
    dataset.addSample( [ 0.20 , 0.80 ], [23] )
    dataset.addSample( [ 0.3  , 1.7  ], [77] )
    dataset.addSample( [ 1.4  , 0.10 ], [77] )

    module  = SVM( dataset.indim, dataset.outdim, SimplePolyKernel() )
    trainer = SVMTrainer(module, cost=200.)
    trainer.setData( dataset )


    from lib.svmtrainer_visualizer import SVMTrainerVisualizer
    SVMTrainerVisualizer(trainer)

    trainer.train()



    wrongcount = 0
    for i in range(dataset.getLength()):
        xi, yi = dataset.getSample(i)
        out = array([[0]])
        module._forwardImplementation( array([xi]), out )
        print " ", i,":   x=", xi, " out=", out[0], " target=", yi[0], "  f = %.2f"%module.rawOutput(xi)
        if out[0] != yi: wrongcount += 1

    print "iterations =", trainer.stepno
    print "samples    =", dataset.getLength()
    print "false_num  =", wrongcount



