#!/usr/bin/python

__author__ = 'Michael Isik'


from pybrain.tools.svmdata             import SVMData
from pybrain.supervised.trainers.mcsvm import MCSVMTrainer
from pybrain.structure.modules.svm     import KT
from pybrain.structure.modules.mcsvm   import MCSVM
from pybrain.tools.gridsearch          import GridSearchCostGamma
from lib.gridsearch_visualizer         import GridSearchVisualizer
import getopt,sys



def getUsage():
    return '''
  USAGE: python gridsearchdoe.py dataset-file
'''


try:
    opts, args = getopt.getopt(sys.argv[1:], "")
except getopt.GetoptError:
    print getUsage()
    exit(2)

if len(args) == 1:
    trainfile = args[0]
else:
    print getUsage()
    exit(2)


dataset = SVMData()
dataset.loadData(trainfile)


module  = MCSVM( dataset.indim, dataset.outdim, KT.RBF )
trainer = MCSVMTrainer( module )


gs = GridSearchCostGamma(trainer, dataset, [-5,-15], [15,3], [5,4], verbose=1)

GridSearchVisualizer(gs)
params = gs.search()
print
print "=== FOUND MAXIMUM AT: ", params























#if __name__ == "__main__":
#    main()
