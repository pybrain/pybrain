from __future__ import print_function

#!/usr/bin/env python
# Example script for feed-forward network usage in PyBrain.
__author__ = "Martin Felder"
__version__ = '$Id$'

from pylab import figure, ioff, clf, contourf, ion, draw, show
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

from .datasets import generateGridData, generateClassificationData, plotData

# load the training data set
trndata = generateClassificationData(250)

# neural networks work better if classes are encoded using
# one output neuron per class
trndata._convertToOneOfMany( bounds=[0,1] )

# same for the independent test data set
tstdata = generateClassificationData(100)
tstdata._convertToOneOfMany( bounds=[0,1] )

# build a feed-forward network with 20 hidden units, plus
# a corresponding trainer
fnn = buildNetwork( trndata.indim, 5, trndata.outdim, outclass=SoftmaxLayer )
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

# generate a grid of data points for visualization
griddata, X, Y = generateGridData([-3.,6.,0.2],[-3.,6.,0.2])

# repeat 20 times
for i in range(20):
    # train the network for 1 epoch
    trainer.trainEpochs( 1 )

    # evaluate the result on the training and test data
    trnresult = percentError( trainer.testOnClassData(),
                              trndata['class'] )
    tstresult = percentError( trainer.testOnClassData(
           dataset=tstdata ), tstdata['class'] )

    # print the result
    print("epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult)

    # run our grid data through the FNN, get the most likely class
    # and shape it into an array
    out = fnn.activateOnDataset(griddata)
    out = out.argmax(axis=1)
    out = out.reshape(X.shape)

    # plot the test data and the underlying grid as a filled contour
    figure(1)
    ioff()  # interactive graphics off
    clf()
    # plot the datapoints
    plotData(tstdata)
    # overlay a contour plot of the functional margin
    if out.max()!=out.min():
        CS = contourf(X, Y, out)
    ion()   # interactive graphics on
    draw()  # update the plot

# show the plot until user kills it
ioff()
show()

