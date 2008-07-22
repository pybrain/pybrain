#!/usr/bin/env python
# Example script for SVM classification using PyBrain and LIBSVM
# CAVEAT: Needs the libsvm Python file svm.py and the corresponding (compiled)
# library to reside in the Python path!

import pylab as p
import logging
from os.path import join
##from matplotlib import axes3d as a3

# load the necessary components
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError

from svmunit                     import SVMUnit
from svmtrainer                  import SVMTrainer
from datagenerator               import generateTwoClassData, plotData, generateGridData

logging.basicConfig(level=logging.INFO, filename=join('.','testrun.log'),
                    format='%(asctime)s %(levelname)s %(message)s')
logging.getLogger('').addHandler(logging.StreamHandler())


# load the training and test data sets
trndata = generateTwoClassData(20)
tstdata = generateTwoClassData(100)

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
trnresult = percentError( svm.forwardPass(dataset=trndata), trndata['target'] )
tstresult = percentError( svm.forwardPass(dataset=tstdata), tstdata['target'] )
print "sigma: %7g,  C: %7g,  train error: %5.2f%%,  test error: %5.2f%%" % (2.0**log2g, 2.0**log2C, trnresult, tstresult)

# generate a grid dataset
griddat, X, Y = generateGridData(x=[-4,8,0.1],y=[-2,3,0.1])

# pass the grid through the SVM, but this time get the raw distance 
# from the boundary, not the class
Z = svm.forwardPass(dataset=griddat, values=True)

# the output format is a bit weird... make it into a decent array
Z = p.array([z.values()[0] for z in Z]).reshape(X.shape)

# this stuff plots a 3d scatterplot
fig = p.figure()
##ax = a3.Axes3D(fig)
##ax.scatter(trndata['input'][:,0],trndata['input'][:,1]) #,color=trndata['target'][:,0].astype(int))
##ax.scatter(X.flatten(),Y.flatten(),Z.flatten())
##ax.set_xlim(-4.,8.)
##ax.set_ylim(-2.,3.)

# use this to save the surface as an ASCII file
##p.save(r'C:\tmp\svmgrid_lg2.txt',Z)

# make a 2d plot
plotData(trndata)
p.contourf(X, Y, Z)
p.show()
