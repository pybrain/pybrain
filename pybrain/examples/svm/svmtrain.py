#!/usr/bin/python

__author__ = 'Michael Isik'

#from pybrain.examples.svm.debug import dbg,pfx,incVerbosity,setVerbosity,enableDebugging
#enableDebugging()
#setVerbosity(4)

from pybrain.datasets import SupervisedDataSet
from pybrain.tools.svmdata import SVMData
from pybrain.supervised.trainers.svm   import SVMTrainer
from pybrain.supervised.trainers.mcsvm import MCSVMTrainer
from pybrain.structure.modules.svm     import KT,SVM,SimplePolyKernel
from pybrain.structure.modules.mcsvm   import MCSVM


from numpy import array

import getopt, sys, scipy

#import profile




def getUsage():
    return '''
  USAGE: python svmtrain.py [options] [training_set_file]

  EXAMPLES:
      python svmtrain.py -c 10 -t 10 -V
      python svmtrain.py -c 10 -t 1 -d 20 -V
      python svmtrain.py -M mymodel.pkl datasets/simple.svm
      python svmtrain

  If training_set_file argument is omitted, a simple 2-dimensional problem
  will be generated.

  OPTIONS:
    -h, --help :  Show help text
    -V         :  Visualize
    -t kernel_type (default 2)
        0  -- Linear                : x1' * x2
        1  -- Polynomial            : ( gamma * x1' * x2 + coef0 ) ^ degree
        2  -- Radial basis function : exp( - gamma * | x1 - x2 | ^ 2 )
        3  -- Sigmoid               : tanh( gamma * x1' * x2 + coef0 )
        10 -- Simple polynomial kernel with explicit feature function phi.
              2D input space, 3D feature space. Use this one for visual
              demonstration of input and feature space. : ( x1' * x2 ) ^ 2
    -d float    :  degree     (default: 3)
    -g float    :  gamma      (default: 1)
    -r float    :  coef0      (default: 0)
    -c float    :  cost value (default: 1)
    -T testfile :  Sample-file to test the learned model on.
    -M dumpfile :  Dump trained module to this file.
'''


def generateSimpleProblem():
    dataset = SupervisedDataSet(2,1)

    dataset.addSample( [ 0.10 , 0.50 ], [ +1 ] )
    dataset.addSample( [ 1.20 , 1.60 ], [ +1 ] )
    dataset.addSample( [ 0.80 , 2.40 ], [ +1 ] )
    dataset.addSample( [ 1.50 , 1.60 ], [ +1 ] )
    dataset.addSample( [ 1.20 , 2.50 ], [ +1 ] )
    dataset.addSample( [ 0.20 , 0.80 ], [ +1 ] )

    dataset.addSample( [ 0.30 , 1.70 ], [ -1 ] )
    dataset.addSample( [ 0.20 , 1.80 ], [ -1 ] )
    dataset.addSample( [ 0.15 , 1.65 ], [ -1 ] )
    dataset.addSample( [ 0.05 , 2.00 ], [ -1 ] )
    dataset.addSample( [ 1.40 , 0.10 ], [ -1 ] )
    dataset.addSample( [ 1.50 , 0.20 ], [ -1 ] )
    dataset.addSample( [ 1.80 , 0.20 ], [ -1 ] )
    dataset.addSample( [ 1.10 , 0.05 ], [ -1 ] )
    return dataset




if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "M:T:hqvt:d:g:r:c:V", ["test-file=", "help", "quiet"])
    except getopt.GetoptError:
        print getUsage()
        sys.exit(2)

    inpfile   = None

    kernel_type = KT.RBF
    degree      = 3.
    gamma       = 1.
    coef0       = 0.
    cost        = 1.
    visualize   = False
    testfile    = None
    dumpfile    = None

    for o, a in opts:
        if o == "-v":
            incVerbosity()
        elif o in ("-t"):
            kernel_type = {
                0:  KT.LINEAR,
                1:  KT.POLY,
                2:  KT.RBF,
                3:  KT.SIGMOID,
                10: SimplePolyKernel()
            }[int(a)]
        elif o in ("-d"): degree = float(a)
        elif o in ("-g"): gamma  = float(a)
        elif o in ("-r"): coef0  = float(a)
        elif o in ("-c"): cost   = float(a)
        elif o in ("-V"):
            visualize = True
        elif o in ("-T","--test-file"):
            testfile = a
        elif o in ("-M"):
            dumpfile = a
        elif o in ("-h", "--help"):
            print getUsage()
            sys.exit()

    trainfile = None
    if len(args):
        trainfile = args[0]



    print "\n=== Loading samples ==="
    multiclass = False
    SVM_        = SVM
    SVMTrainer_ = SVMTrainer
    if trainfile:
        print "Loading from file:",trainfile
        dataset = SVMData()
        dataset.loadData(trainfile)

        if dataset.nClasses > 2:
            multiclass = True
            SVM_        = MCSVM
            SVMTrainer_ = MCSVMTrainer
        print "Class histogram"
        for label,count in dataset.getClassHistogram().items():
            print "  ",label," :",count

    else:
        print "Generating samples"
        dataset = generateSimpleProblem()


    l = dataset.getLength()

    module  = SVM_( dataset.indim, dataset.outdim, kernel_type, degree=degree, gamma=gamma, coef0=coef0 )

    if not multiclass:
        print "\n=== Active configuration ==="
        kernel = module._kernel
        print "kernel class =", kernel.__class__
        print "degree       =", kernel._degree
        print "gamma        =", kernel._gamma
        print "coef0        =", kernel._coef0
        print "cost         =", cost

    print "\n=== Initializing Trainer ==="
    trainer = SVMTrainer_(module, cost=cost)
    trainer.setData(dataset)



    if visualize:
        print "\n=== Initializing visualization ==="
        if not multiclass:
            from lib.svmtrainer_visualizer import SVMTrainerVisualizer
            SVMTrainerVisualizer(trainer)
        else:
            print "Visualization not possible in multiclass mode"
            print "Skipping"



    print "\n=== Entering training loop ==="
#    profile.run("trainer.train()");exit(0)
    trainer.train()
#    exit()

    if dumpfile:
        print "\n=== Dumping model to file: \"%s\" ==="%dumpfile
        module.dumpToFile(dumpfile)

    if testfile:
        testset = SVMData(testfile)
    else:
        testset = dataset

    print "\n=== Testing ==="
    wrongcount = 0
    for i in xrange(testset.getLength()):
        xi,yi = testset.getSample(i)
        out = array([[0]])
        try:
            module._forwardImplementation( array([xi]), out )
#        print "----->      x=", xi,
#        print " out=", out[0], " target=", yi, "     <------"
            if out[0] != yi: wrongcount += 1
        except Exception: pass


    if not multiclass:
        print "iterations =", trainer.stepno
    print "samples    =", l
    print "false_num  =", wrongcount


