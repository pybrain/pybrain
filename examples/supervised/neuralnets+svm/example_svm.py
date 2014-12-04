from __future__ import print_function

#!/usr/bin/env python
""" Example script for SVM classification using PyBrain and LIBSVM
CAVEAT: Needs the libsvm Python file svm.py and the corresponding (compiled) library to reside in the Python path! """

__author__ = "Martin Felder"
__version__ = '$Id$'

import pylab as p
import logging
from os.path import join

# load the necessary components
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError

from pybrain.structure.modules.svmunit        import SVMUnit
from pybrain.supervised.trainers.svmtrainer   import SVMTrainer

# import some local stuff
from .datasets               import generateClassificationData, plotData, generateGridData

logging.basicConfig(level=logging.INFO, filename=join('.','testrun.log'),
                    format='%(asctime)s %(levelname)s %(message)s')
logging.getLogger('').addHandler(logging.StreamHandler())


# load the training and test data sets
trndata = generateClassificationData(20, nClasses=2)
tstdata = generateClassificationData(100, nClasses=2)

# initialize the SVM module and a corresponding trainer
svm = SVMUnit()
trainer = SVMTrainer( svm, trndata )

# train the with fixed meta-parameters
log2C=0.   # degree of slack
log2g=1.1  # width of RBF kernels
trainer.train( log2C=log2C, log2g=log2g )
# alternatively, could train the SVM using design-of-experiments grid search
##trainer.train( search="GridSearchDOE" )

# pass data sets through the SVM to get performance
trnresult = percentError( svm.activateOnDataset(trndata), trndata['target'] )
tstresult = percentError( svm.activateOnDataset(tstdata), tstdata['target'] )
print("sigma: %7g,  C: %7g,  train error: %5.2f%%,  test error: %5.2f%%" % (2.0**log2g, 2.0**log2C, trnresult, tstresult))

# generate a grid dataset
griddat, X, Y = generateGridData(x=[-4,8,0.1],y=[-2,3,0.1])

# pass the grid through the SVM, but this time get the raw distance
# from the boundary, not the class
Z = svm.activateOnDataset(griddat, values=True)

# the output format is a bit weird... make it into a decent array
Z = p.array([list(z.values())[0] for z in Z]).reshape(X.shape)

# make a 2d plot of training data with an decision value contour overlay
fig = p.figure()
plotData(trndata)
p.contourf(X, Y, Z)
p.show()
