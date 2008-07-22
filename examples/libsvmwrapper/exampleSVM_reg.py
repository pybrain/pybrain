#!/usr/bin/env python
# Example script for SVM classification using PyBrain and LIBSVM
# CAVEAT: Needs the libsvm Python file svm.py and the corresponding (compiled)
# library to reside in the Python path!

import pylab as p
import logging
from os.path import join
import dislin

# load the necessary components
from pybrain.datasets            import SequentialDataSet
from pybrain.utilities           import percentError
from lib.svmunit                 import SVMUnit
from lib.svmtrainer              import SVMTrainer
from datagenerator               import generateGridData

logging.basicConfig(level=logging.INFO, filename=join('.','testrun.log'),
                    format='%(asctime)s %(levelname)s %(message)s')
logging.getLogger('').addHandler(logging.StreamHandler())

dat = p.load('body_fat.txt',skiprows=15)

# load the training and test data sets
trndata = SequentialDataSet(13,1)
trndata.setField('input',dat[:,2:])
trndata.setField('target',dat[:,1])
#plotData(trndata)
#p.show()
# initialize the SVM module and a corresponding trainer
svm = SVMUnit()
trainer = SVMTrainer( svm, trndata, svm_type=3 )

# train the SVM design-of-experiments grid search
#log2g = 
trainer.train( log2C=0., log2g=2.) #search="GridSearchDOE" )

# pass data sets through the SVM to get performance
trnresult = percentError( svm.forwardPass(dataset=trndata), trndata['target'] )
tstresult = percentError( svm.forwardPass(dataset=tstdata), tstdata['target'] )
#print "sigma: %7g,  C: %7g,  train error: %5.2f%%,  test error: %5.2f%%" % (trnresult, "tstresult

#rawout = array(U.forwardPass(dataset=DS, values=True))
griddat, X, Y = generateGridData(x=[-4,8,0.1],y=[-2,3,0.1])
Z = svm.forwardPass(dataset=griddat, values=True)
Z = p.array([z.values()[0] for z in Z]).reshape(X.shape)
fig = p.figure()
ax = a3.Axes3D(fig)
ax.scatter(trndata['input'][:,0],trndata['input'][:,1]) #,color=trndata['target'][:,0].astype(int))
ax.scatter(X.flatten(),Y.flatten(),Z.flatten())
ax.set_xlim(-4.,8.)
ax.set_ylim(-2.,3.)

#p.save(r'C:\tmp\svmgrid_lg2.txt',Z)
#a3.a
#plotData(trndata)
#p.contourf(X, Y, Z)
p.show()
