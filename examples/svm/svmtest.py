#!/usr/bin/python

__author__ = 'Michael Isik'


from pybrain.structure.modules.svm import SVM
from pybrain.tools.svmdata import SVMData
from numpy import sum
import getopt,sys


def getUsage():
    return '''
  USAGE: python svmtest.py model_file testset_file
'''


if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "")
    except getopt.GetoptError:
        print getUsage()
        exit(2)

    if len(args) == 2:
        dumpfile = args[0]
        testfile = args[1]
    else:
        print getUsage()
        exit(2)


    print "\n=== Loading samples ==="
    testset = SVMData(testfile)


    print "\n=== Loading module ==="
#    module = SVM( testset.indim, testset.outdim ).loadFromFile(dumpfile)
    module = SVM.loadFromFile(dumpfile)


    print "Class histogram"
    for label,count in testset.getClassHistogram().items():
        print "  ",label," :",count

    l = testset.getLength()
    print "\n=== Testing ==="
    wrongcount = 0
#    out = empty((l,1),float)
#    module._forwardImplementation( testset.getField("input"), out )
    out = module.classify( testset.getField("input") )
    wrong = testset.getField("target").flatten() != out

#    for i in xrange(l):
#        xi,yi = testset.getSample(i)
#        out = [0]
#        module._forwardImplementation( [xi], out )
#        if out[0] != yi: wrongcount += 1


    print "samples    =", l
    print "false_num  =", sum(wrong)


