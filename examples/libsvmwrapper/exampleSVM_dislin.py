#!/usr/bin/env python
# Example script for SVM classification using PyBrain and LIBSVM
# CAVEAT: Needs the libsvm Python file svm.py and the corresponding (compiled)
# library to reside in the Python path!

import pylab as p
import logging
from os.path import join
import dislin

# load the necessary components
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from lib.svmunit                 import SVMUnit
from lib.svmtrainer              import SVMTrainer
from datagenerator               import generateTwoClassData, plotData, generateGridData

logging.basicConfig(level=logging.INFO, filename=join('.','testrun.log'),
                    format='%(asctime)s %(levelname)s %(message)s')
logging.getLogger('').addHandler(logging.StreamHandler())


# load the training and test data sets
trndata = generateTwoClassData(20)
tstdata = generateTwoClassData(100)
#plotData(trndata)
#p.show()
# initialize the SVM module and a corresponding trainer
svm = SVMUnit()
trainer = SVMTrainer( svm, trndata )

# train the SVM design-of-experiments grid search
#log2g = 
trainer.train( log2C=0., log2g=2.) #search="GridSearchDOE" )
2
# pass data sets through the SVM to get performance
trnresult = percentError( svm.forwardPass(dataset=trndata), trndata['target'] )
tstresult = percentError( svm.forwardPass(dataset=tstdata), tstdata['target'] )
#print "sigma: %7g,  C: %7g,  train error: %5.2f%%,  test error: %5.2f%%" % (trnresult, "tstresult

#rawout = array(U.forwardPass(dataset=DS, values=True))
griddat, x, y = generateGridData(x=[-4,8,0.1],y=[-2,3,0.1], return_ticks=True)
Z = svm.forwardPass(dataset=griddat, values=True)
Z = p.array([z.values()[0] for z in Z]).reshape([y.size,x.size])

dislin.scrmod ('revers')
dislin.metafl ('cons')
dislin.setpag ('da4p')
dislin.disini ()
dislin.pagera ()
dislin.hwfont ()

dislin.titlin ('SVM surface', 1)

dislin.axspos (200, 2600)
dislin.axslen (1800, 1800)

dislin.name   ('X-axis', 'X')
dislin.name   ('Y-axis', 'Y')
dislin.name   ('Z-axis', 'Z')

axspec = {"x low":-4, "x high":8., "x label0":-4., "x labstep":2,
          "y low":-2, "y high":3., "y label0":-2., "y labstep":1,
          "z low":-1.5, "z high":1.5, "z label0":-1.5, "z labstep":1.5}
dislin.view3d (20,-12,2,'USER')
dislin.graf3d (-4,8,-4,2, -2,3,-2,1, -1.5,1.5,-1.5,1.5)
dislin.height (50)
dislin.title  ()

dislin.grfini (-1., -1., 0.0, 1., -1., 0.0, 1., 1., 0.0)
dislin.nograf ()
dislin.graf (-4,8,-4,2, -2,3,-2,1)
  

zlev = p.arange(-1.5,1.5,0.2)
#dislin.qplsca(trndata['input'][:,0].flatten(),trndata['input'][:,1].flatten(),trndata['input'][:,1].size)
for cl in zlev:
    dislin.contur (x, x.size, y, y.size, Z.transpose(), cl)#, zlev.size)

dislin.box2d ();
dislin.reset ('nograf');
dislin.grffin ();

dislin.shdmod ('smooth', 'surface')
dislin.light('ON')
dislin.surshd (x, x.size, y, y.size, Z.transpose())
dislin.disfin ()
