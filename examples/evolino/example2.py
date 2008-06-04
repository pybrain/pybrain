#!/usr/bin/env python

__author__ = 'Michael Isik'

# Example script for lstm network usage in PyBrain in combination with Evolino.

# load the necessary components
from pybrain.datasets.sequential import SequentialDataSet
from pybrain.datasets.classification import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.supervised.trainers.evolino import EvolinoTrainer
from pybrain.tools.validation import testOnSequenceData
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.tools.svmdata import SVMData
from pybrain.structure.modules   import LSTMLayer

from numpy import array, zeros




def loadDataset(filename):
    raw_dataset = SVMData()
    raw_dataset.loadData(filename)
    dataset = SequentialDataSet(4,3)
    for inp, tar in raw_dataset:
        new_tar = zeros(3)
        new_tar[tar[0]] = 1
        dataset.newSequence()
        n_samples = len(inp)/4
        for i in range(n_samples):
            base = i*4
            new_inp = array([
                inp[base+0],
                inp[base+1],
                inp[base+2],
                inp[base+3]
                ])
            dataset.addSample( new_inp , new_tar )
    return dataset


# load the training data set, will be in oneOfMany format
trndata=loadDataset('/michael/svm/vonmartin/evolinotest/finger0av_10s_20w_scale.svm')

# same for the independent test data set
tstdata=loadDataset('/michael/svm/vonmartin/evolinotest/finger0av_10s_20w_scale.svm')

# build a lstm network with 20 hidden units, plus a corresponding trainer
net = buildNetwork( trndata.indim, 20, trndata.outdim, hiddenclass=LSTMLayer, outputbias=False )
trainer = EvolinoTrainer(
    net,
    dataset=trndata,
    sub_population_size=20,
    evalfunc=testOnSequenceData,
    verbosity=2)


# repeat 5 times
for i in range(100):
    # train the network for 1 epoch
    trainer.trainEpochs( 1 )

    # evaluate the result on the training and test data
    trnresult = testOnSequenceData(net, trndata)*100.
    tstresult = testOnSequenceData(net, tstdata)*100.

    # print the result
    print "epoch: %4d   train error: %5.2f%%   test error: %5.2f%%" % (trainer.totalepochs, trnresult, tstresult)



