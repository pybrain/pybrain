from __future__ import print_function

#!/usr/bin/env python


__author__ = 'Michael Isik'

from pylab import plot, show, ion, cla, subplot, title, figlegend, draw
import numpy

from pybrain.structure.modules.evolinonetwork import EvolinoNetwork
from pybrain.supervised.trainers.evolino      import EvolinoTrainer
from lib.data_generator import generateSuperimposedSineData

print()
print("=== Learning to extrapolate 5 superimposed sine waves ===")
print()
sinefreqs = ( 0.2, 0.311, 0.42, 0.51, 0.74 )
# sinefreqs = ( 0.2, 0.311, 0.42, 0.51, 0.74, 0.81 )
metascale = 8.



scale    = 0.5 * metascale
stepsize = 0.1 * metascale


# === create training dataset
# the sequences must be stored in the target field
# the input field will be ignored
print("creating training data")
trnInputSpace = numpy.arange( 0*scale , 190*scale , stepsize )
trnData = generateSuperimposedSineData(sinefreqs, trnInputSpace)

# === create testing dataset
print("creating test data")
tstInputSpace = numpy.arange( 400*scale , 540*scale , stepsize)
tstData = generateSuperimposedSineData(sinefreqs, tstInputSpace)



# === create the evolino-network
print("creating EvolinoNetwork")
net = EvolinoNetwork( trnData.outdim, 40 )



wtRatio = 1./3.

# === instantiate an evolino trainer
# it will train our network through evolutionary algorithms
print("creating EvolinoTrainer")
trainer = EvolinoTrainer(
    net,
    dataset=trnData,
    subPopulationSize = 20,
    nParents = 8,
    nCombinations = 1,
    initialWeightRange = ( -0.01 , 0.01 ),
#    initialWeightRange = ( -0.1 , 0.1 ),
#    initialWeightRange = ( -0.5 , -0.2 ),
    backprojectionFactor = 0.001,
    mutationAlpha = 0.001,
#    mutationAlpha = 0.0000001,
    nBurstMutationEpochs = numpy.Infinity,
    wtRatio = wtRatio,
    verbosity = 2)



# === prepare sequences for extrapolation and plotting
trnSequence = trnData.getField('target')
separatorIdx = int(len(trnSequence)*wtRatio)
trnSequenceWashout = trnSequence[0:separatorIdx]
trnSequenceTarget  = trnSequence[separatorIdx:]

tstSequence = tstData.getField('target')
separatorIdx = int(len(tstSequence)*wtRatio)
tstSequenceWashout = tstSequence[0:separatorIdx]
tstSequenceTarget  = tstSequence[separatorIdx:]


ion() # switch matplotlib to interactive mode
for i in range(3000):
    print("======================")
    print("====== NEXT RUN ======")
    print("======================")

    print("=== TRAINING")
    # train the network for 1 epoch
    trainer.trainEpochs( 1 )


    print("=== PLOTTING\n")
    # calculate the nets output for train and the test data
    trnSequenceOutput = net.extrapolate(trnSequenceWashout, len(trnSequenceTarget))
    tstSequenceOutput = net.extrapolate(tstSequenceWashout, len(tstSequenceTarget))

    # plot training data
    sp = subplot(211) # switch to the first subplot
    cla() # clear the subplot
    title("Training Set") # set the subplot's title
    sp.set_autoscale_on( True ) # enable autoscaling
    targetline = plot(trnSequenceTarget,"r-") # plot the targets
    sp.set_autoscale_on( False ) # disable autoscaling
    outputline = plot(trnSequenceOutput,"b-") # plot the actual output


    # plot test data
    sp = subplot(212)
    cla()
    title("Test Set")
    sp.set_autoscale_on( True )
    plot(tstSequenceTarget,"r-")
    sp.set_autoscale_on( False )
    plot(tstSequenceOutput,"b-")

    # create a legend
    figlegend((targetline, outputline),('target','output'),('upper right'))

    # draw everything
    draw()


show()




