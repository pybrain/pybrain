############################################################################
# PyBrain Tutorial "Classification with Feed-Forward Neural Networks"
# 
# Author: Martin Felder, felder@in.tum.de
############################################################################

""" This tutorial walks you through the process of setting up a dataset
for classification, and train a network on it while visualizing the results
online. 

First we need to import the necessary components from PyBrain."""

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

""" Furthermore, pylab is needed for the graphical output. """
import pylab as p
import numpy

from examples.neuralnets.datagenerator import generateGridData, plotData

""" To have a nice dataset for visualization, we produce a set of 
points in 2D belonging to three different classes. You could also
read in your data from a file, e.g. using pylab.load(). """

means = [(-1,0),(2,4),(3,1)]
cov = [p.diag([1,1]), p.diag([0.5,1.2]), p.diag([1.5,0.7])]
alldata = ClassificationDataSet(2, 1, nb_classes=3)
for n in xrange(400):
    for klass in range(3):
        input = numpy.random.multivariate_normal(means[n],cov[n])
        alldata.addSample(input, [klass])

""" Randomly split the dataset into 75% training and 25% test data sets. Of course, we
could also have created two different datasets to begin with.""" 
tstdata, trndata = alldata.splitWithProportion( 0.25 )

""" For neural network classification, it is highly advisable to encode classes 
with one output neuron per class. Note that this operation duplicates the original
targets and stores them in an (integer) field named 'class'."""
trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

""" Test our dataset by printing a little information about it. """
print "Number of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0], trndata['class'][0] 

""" Now build a feed-forward network with 5 hidden units. We use the a convenience
function for this. The input and output
layer size must match the dataset's input and target dimension. You could add 
additional hidden layers by inserting more numbers giving the desired layer sizes.
The output layer uses a softmax function because we are doing classification. 
There are more options to explore here, e.g. try changing the hidden layer transfer
function to linear instead of (the default) sigmoid. 

.. seealso:: Desciption :func:`buildNetwork` for more info on options, 
   and the Network tutorial :ref:`netmodcon` for info on how to build 
   your own non-standard networks. 

"""
fnn = buildNetwork( trndata.indim, 5, trndata.outdim, outclass=SoftmaxLayer )

""" 
.. note:: TODO: improve following text

a corresponding trainer """
#trainer = RPropMinusTrainer( fnn, dataset=trndata, verbose=True, weightdecay=0.0)
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

""" generate a grid of data points for visualization """
griddata, X, Y = generateGridData(-3.,6.,0.2)

""" repeat 20 times """
for i in range(20):

    """ train the network for 1 epoch """
    trainer.trainEpochs( 1 )
    
    """ evaluate the result on the training and test data """
    trnresult = percentError( trainer.testOnClassData(), 
                              trndata['class'] )
    tstresult = percentError( trainer.testOnClassData( 
           dataset=tstdata ), tstdata['class'] )

    print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult

    """ run our grid data through the FNN, get the most likely class 
    and shape it into an array """
    out = fnn.activateOnDataset(griddata)
    out = out.argmax(axis=1)
    out = out.reshape(X.shape)
    
    """ plot the test data and the underlying grid as a filled contour """
    p.figure(1)
    p.ioff()  # interactive graphics off
    p.clf()
    plotData(tstdata)
    if out.max()!=out.min():
        CS = p.contourf(X, Y, out)
    p.ion()   # interactive graphics on
    p.draw()  # update the plot
    
""" show the plot until user kills it """
p.ioff()
p.show()  
