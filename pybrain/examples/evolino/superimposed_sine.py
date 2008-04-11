

__author__ = 'Michael Isik'


#from pybrain.datasets.supervised         import SupervisedDataSet
from pybrain.datasets.sequential         import SequentialDataSet
from pybrain.supervised.trainers.evolino import EvolinoTrainer
from pybrain.tools.shortcuts             import buildNetwork
from pybrain.structure.modules           import LSTMLayer
from pybrain.tools.shortcuts             import buildNetwork
from pybrain.tools.validation            import ModuleValidator


from numpy import array, arange, sin, apply_along_axis

class SuperimposedSine(object):
    def __init__(self, lambdas=[1.]):
        self.lambdas = array(lambdas, float)

    def getFuncValue(self, x):
        val = 0.
        for l in self.lambdas:
            val += sin(x*l)
        return val

    def getFuncValues(self, x_array):
        return apply_along_axis(self.getFuncValue, 0, x_array)


def addSequence(dataset, input, target):
    dataset.newSequence()
    for i in xrange(len(input)):
        dataset.addSample(input[i], target[i])
    

sine = SuperimposedSine()
#sine = SuperimposedSine( ( 0.2, 0.311, 0.42, 0.51, 0.74 ) )


trn_data = SequentialDataSet(1,1)
tst_data = SequentialDataSet(1,1)

tst_input = arange( 50 , 60 , 0.3 )
tst_target = sine.getFuncValues(tst_input)
addSequence(tst_data, tst_input, tst_target)

trn_intervals = [(0,4), (6,16),(24,38), (35,39), (40,45)]
for interval in trn_intervals:
    print interval
    trn_input = arange( interval[0], interval[1], 0.3 )
    trn_target = sine.getFuncValues(trn_input)
    addSequence(trn_data, trn_input, trn_target)

trn_input = trn_data.getField('input')
trn_target= trn_data.getField('target')

#for i in xrange(len(trn_input)):
#    trn_data.addSample(trn_input[i], trn_target[i])

#for i in xrange(len(tst_input)):
#    tst_data.addSample(tst_input[i], tst_target[i])




net = buildNetwork( trn_data.indim, 10, trn_data.outdim, hiddenclass=LSTMLayer, outputbias=False )
trainer = EvolinoTrainer(
    net,
    dataset=trn_data,
    sub_population_size = 20,
    mutation_alpha = 0.01,
    initial_weight_range = ( -0.5 , 0.5 ),
    verbosity = 2)


from pylab import plot, show, ion, cla
ion()
#show()


for i in range(300):
    # train the network for 1 epoch
    cla()
    trainer.trainEpochs( 1 )

    # evaluate the result on the training and test data
    trn_result = -ModuleValidator.MSE(net, trn_data)
    tst_result = -ModuleValidator.MSE(net, tst_data)
#   trn_result = testOnSequenceData(net, trn_data)*100.
#   tst_result = testOnSequenceData(net, tst_data)*100.
    trn_output = ModuleValidator.calculateModuleOutput(net, trn_data)
    tst_output = ModuleValidator.calculateModuleOutput(net, tst_data)
    plot(trn_input,trn_output,"o")
    plot(trn_input,trn_target,"o")
    plot(tst_input,tst_output,"o")
    plot(tst_input,tst_target,"o")

    # print the result
    print "epoch: %4d   train performance: %5.2f   test performance: %5.2f" % (trainer.totalepochs, trn_result, tst_result)


show()




