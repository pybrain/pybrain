#!/usr/bin/env python
__author__ = "Martin Felder"
__version__ = '$Id$' 

from pylab import plot, hold, show #@UnresolvedImport
from scipy import sin, rand, arange
from pybrain.datasets            import SequenceClassificationDataSet
from pybrain.structure.modules   import LSTMLayer, SoftmaxLayer
from pybrain.supervised          import RPropMinusTrainer
from pybrain.tools.validation    import testOnSequenceData
from pybrain.tools.shortcuts     import buildNetwork

def dummyData( npoints, nseq, noise=0.3 ):
    """ construct a 2-class dummy dataset out of noisy sines """
    x = arange(npoints)/float(npoints) * 20.
    y1 = sin(x+rand(1)*3.)
    y2 = sin(x/2.+rand(1)*3.)
    DS = SequenceClassificationDataSet(1,1, nb_classes=2)
    for _ in xrange(nseq):
        DS.newSequence()
        buf = rand(npoints)*noise + y1 + (rand(1)-0.5)*noise
        for i in xrange(npoints):
            DS.addSample([buf[i]],[0])
        DS.newSequence()
        buf = rand(npoints)*noise + y2 + (rand(1)-0.5)*noise
        for i in xrange(npoints):
            DS.addSample([buf[i]],[1])
    return DS


# create training and test data
trndata = dummyData(50, 40)
trndata._convertToOneOfMany( bounds=[0.,1.] )
tstdata = dummyData(50, 20)
tstdata._convertToOneOfMany( bounds=[0.,1.] )

# plot the first 5 timeseries
plot(trndata['input'][0:250,:],'-o')
hold(True)
plot(trndata['target'][0:250,0])

# TODO: This blocks the rest of the execution! FIX!
show()

# construct LSTM network - note the missing output bias
rnn = buildNetwork( trndata.indim, 5, trndata.outdim, hiddenclass=LSTMLayer, outclass=SoftmaxLayer, outputbias=False, recurrent=True)

#trainer = BackpropTrainer( rnn, dataset=trndata, verbose=True, momentum=0.9, learningrate=0.00001 )
trainer = RPropMinusTrainer( rnn, dataset=trndata, verbose=True )
for i in xrange(100):
    #sortedProfiling('trainer.trainEpochs(5)')
    trainer.trainEpochs( 2 )
    trnresult = 100. * (1.0-testOnSequenceData(rnn, trndata))
    tstresult = 100. * (1.0-testOnSequenceData(rnn, tstdata))
    print "train error: %5.2f%%" % trnresult, ",  test error: %5.2f%%" % tstresult
