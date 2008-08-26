#!/usr/bin/python

__author__ = 'Michael Isik'


from pybrain.tools.svmdata import SVMData
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers.mcsvm import MCSVMTrainer
from pybrain.structure.modules.svm import KT,MCSVM
import getopt,sys


def getUsage():
    return '''
  USAGE: python mctests.py trainingset_file testset_file
'''


if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "")
    except getopt.GetoptError:
        print getUsage()
        exit(2)

    if len(args) == 2:
        trainfile = args[0]
        testfile  = args[1]
    else:
        print getUsage()
        exit(2)


    dataset = SVMData()
    dataset.loadData(trainfile)



    module  = MCSVM( dataset.indim, dataset.outdim, KT.RBF )
    trainer = MCSVMTrainer(module, cost=200.)
    trainer.setData( dataset )
    trainer.train()





    meta_ds = SupervisedDataSet( len(module._sub_modules), 1 )
    for i in range( dataset.getLength() ):
        xi, yi = dataset.getSample(i)
        margins = module.rawOutputs(xi)
        meta_ds.addSample( margins, yi )

    meta_module  = MCSVM( meta_ds.indim, meta_ds.outdim, KT.RBF )
    meta_trainer = MCSVMTrainer(meta_module, cost=2000.)
    meta_trainer.setData( meta_ds )
    meta_trainer.train()



    dataset = SVMData()
    dataset.loadData(testfile)

    wrongcount = 0
    meta_wrongcount = 0
    for i in range(dataset.getLength()):
        xi, yi = dataset.getSample(i)
        out = module.classify( xi )
        margins = module.rawOutputs(xi),
        meta_out = meta_module.classify( margins )
#        print " ", i,":   x =", xi, " out =", out[0], " target =", yi[0], module.rawOutputs(xi)
        print " ", i,":   out =", out, " meta_out = ", meta_out, " target =", yi[0],
        if out != yi:
            print "   <<< FALSE",
            wrongcount += 1
        if meta_out != yi:
            print "   <<< META_FALSE",
            meta_wrongcount += 1
        print

    print "samples    =", dataset.getLength()
    print "wrong      =", wrongcount
    print "meta_wrong =", meta_wrongcount



