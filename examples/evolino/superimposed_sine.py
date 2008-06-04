

__author__ = 'Michael Isik'


#from pybrain.datasets.supervised         import SupervisedDataSet
from pybrain.datasets.sequential         import SequentialDataSet
from pybrain.supervised.trainers.evolino import EvolinoTrainer
from pybrain.tools.shortcuts             import buildNetwork
from pybrain.structure.modules           import LSTMLayer
from pybrain.tools.shortcuts             import buildNetwork
from pybrain.tools.validation            import ModuleValidator
from pybrain.rl.learners.blackboxoptimizers.evolino.networkwrapper import EvolinoNetwork # gehoert in zu den modulen
from pylab import axes, plot, show, ion, cla, subplot, figure
import pylab

#import time
#from pprint import pprint
from numpy import array, arange, sin, apply_along_axis, append, zeros

class SuperimposedSine(object):
    """ Small class for generating superimposed sine signals
    """
    def __init__(self, lambdas=[1.]):
        self.lambdas = array(lambdas, float)

    def getFuncValue(self, x):
        val = 0.
        for l in self.lambdas:
            val += sin(x*l)
        return val

    def getFuncValues(self, x_array):
        return apply_along_axis(self.getFuncValue, 0, x_array)


def addSequence(dataset, target):
    """ Extracts the elements of the input and target arrays, and adds them
        to dataset as a new sequence.
    """
    dataset.newSequence()
    for i in xrange(len(target)):
        dataset.addSample([], target[i])


#sine = SuperimposedSine()
# create a superimposed sine object with 6 overlayed sine waves
#sine = SuperimposedSine( ( 0.2, 0.311, 0.42, 0.51, 0.74, 1 ) )
#sine = SuperimposedSine( ( 1.6, 3.8, 5 ) )
sine = SuperimposedSine( ( 1.6, 3.8, 5 ) )
#sine = SuperimposedSine( ( 1.6, 3.8, 5, 6 ) )
#sine = SuperimposedSine( ( 2.6, 4.8, 6, 7 ) )
#sine = SuperimposedSine( ( 1.6, 3.8 ) )
#sine = SuperimposedSine( [ 1.4 ] )


trn_data = SequentialDataSet(0,1)
tst_data = SequentialDataSet(0,1)


# create the training dataset
#trn_input_space = arange( 0 , 31 , 1.0 )
trn_input_space = arange( 0 , 90 , 0.2 )
trn_target = sine.getFuncValues(trn_input_space)
#trn_input = append( [0], trn_target[0:-1] )
#trn_input = zeros(len(trn_target))
addSequence(trn_data, trn_target)


# create the testing dataset
tst_input_space = arange( 300 , 440 , 0.2 )
tst_target = sine.getFuncValues(tst_input_space)
#tst_input = append( [0], tst_target[0:-1] )
#trn_input = zeros(len(tst_target))
addSequence(tst_data, tst_target)


# build a network with a hidden lstm layer
#net = buildNetwork( trn_data.indim, 6, trn_data.outdim, hiddenclass=LSTMLayer, outputbias=False )
net = EvolinoNetwork( trn_data.indim, trn_data.outdim, 20 )

#wtv_ratio = (1,2,1)
#wc_ratio  = (3,1)
wtv_ratio = (1,3,2)
wc_ratio  = (5,2)


# instantiate an evolino trainer, that will train our network through evolutionary algorithms
trainer = EvolinoTrainer(
    net,
    dataset=trn_data,
    subPopulationSize = 4,
    nCombinations = 10,
    initialWeightRange = ( -0.01 , 0.01 ),
#    initialWeightRange = ( -0.0 , 0.0 ),
    backprojectionFactor = 0.01,
    mutationAlpha = 0.01,
    nBurstMutationEpochs = 20,
    wtvRatio = wtv_ratio,
    verbosity = 2)


ion() # switch matplotlib to interactive mode


for i in range(300):
    print "======================"
    print "====== NEXT RUN ======"
    print "======================"

    print "=== TRAINING"
    # train the network for 1 epoch
    trainer.trainEpochs( 1 )
#    exit()
    # evaluate the result on the training and test data
    print "=== EVALUATING"
    print "Network weights:"

    # print out the weights of the network
    for chromosome in trainer.network.getGenome():
        # dirty hack, who cares
        print array(array(array(chromosome)*1000,int),float) / 1000
    print


    # calculate the MSE on the training and dataset
    trn_result = -ModuleValidator.MSE(net, trn_data)
    tst_result = -ModuleValidator.MSE(net, tst_data)

    # print the results
    print "STATS: Epoch: %4d  Train Performance: %5.4f   Test Performance: %5.4f" % (trainer.totalepochs, trn_result, tst_result)
    print


    print "=== PLOTTING\n"
    # calculate the nets output on the training and the testing dataset
#    trn_output = ModuleValidator.calculateOutput(net, trn_data)
#    tst_output = ModuleValidator.calculateOutput(net, tst_data)
    trn_input, trn_output, trn_target = net.calculateOutput(trn_data, wc_ratio)
#    print trn_output
#    print
#    print trn_target
#    print
#    print trn_input
#    exit()
    tst_input, tst_output, tst_target = net.calculateOutput(tst_data, wc_ratio)

    sp = subplot(211) # switch to the first subplot
    cla() # clear the subplot
    pylab.title("Training Set") # set the subplot's title
    sp.set_autoscale_on( True ) # enable autoscaling
    targetline = plot(trn_target,"r-") # plot the targets
    sp.set_autoscale_on( False ) # disable autoscaling
    outputline = plot(trn_output,"b-") # plot the actual output


    # do the same stuff to the second subplot, except that testing data is used
    sp = subplot(212)
    cla()
    pylab.title("Test Set")
    sp.set_autoscale_on( True )
    plot(tst_target,"r-")
    sp.set_autoscale_on( False )
    plot(tst_output,"b-")

    # create a legend
    pylab.figlegend((targetline, outputline),('target','output'),('upper right'))

    # draw everything
    pylab.draw()


show()




