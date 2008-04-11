

__author__ = 'Michael Isik'


from pybrain.structure.networks.network     import Network
from pybrain.structure.modules.tanhlayer    import TanhLayer
from pybrain.structure.modules.linearlayer  import LinearLayer
from pybrain.structure.modules.biasunit     import BiasUnit
from pybrain.structure.connections.full     import FullConnection
from pybrain.structure.connections.identity import IdentityConnection
from pybrain.supervised.trainers.backprop   import BackpropTrainer
from pybrain.datasets.importance            import ImportanceDataSet
from pybrain.structure.modules.lstm         import LSTMLayer
#from pybrain.tools.validation               import Validator, ModuleValidator, SequenceHelper
from pybrain.tools.validation               import testOnData2


from pprint import pprint
#from pybrain.rl.learners.blackboxoptimizers.evolino.evolino import Evolino
from pybrain.supervised.trainers.evolino import EvolinoTrainer


from pybrain.tools.svmdata import SVMData
from pprint import pprint
from scipy import array, zeros

def main():
    dataset = getSimpleDataset()
    dataset = getDataset()

    net = Network()
    indim,outdim = dataset.indim, dataset.outdim
    in_layer  = LinearLayer(indim)
#    hid_layer = LSTMLayer(indim)
    hid_layer = LSTMLayer(30)
    out_layer = LinearLayer(outdim)
    bias      = BiasUnit()

    net.addInputModule( in_layer )
    net.addModule(hid_layer)
    net.addModule(bias)
    net.addOutputModule(out_layer)

    net.addConnection( FullConnection( in_layer  , hid_layer ) )
    net.addConnection( FullConnection( hid_layer , out_layer ) )
    net.addConnection( FullConnection( bias, hid_layer ) )
#    net.addConnection( FullConnection( bias, out_layer ) )
#    net.sortModules()

    trainer = EvolinoTrainer(
        net,
        dataset=dataset,
        sub_population_size=20,
        evalfunc=testOnData2,
        verbosity=2)

    for i in range(100):
        trainer.trainEpochs(1)

        performance = testOnData2(net, dataset)
        print "performance:", performance




def getDataset():
    svmdataset = SVMData()
    svmdataset.loadData("/michael/svm/vonmartin/evolinotest/finger0av_10s_20w_scale.svm")
    dataset = ImportanceDataSet(4,3)
    for inp, tar in svmdataset:
#        print inp
#        print tar
        new_tar = zeros(3)
        new_tar[tar[0]] = 1
        dataset.newSequence()
        print
        n_samples = len(inp)/4
        for i in range(n_samples):
            base = i*4
            new_inp = array([
                inp[base+0],
                inp[base+1],
                inp[base+2],
                inp[base+3]
                ])
            if i == n_samples-1: importance = 1
            else: importance = 1
#            scaled_tar = new_tar / (n_samples-i)
#            print scaled_tar
            dataset.addSample( new_inp , new_tar, importance )
#            print new_inp, new_tar
    return dataset

def getSimpleDataset():
    dataset = ImportanceDataSet(2,1)
    dataset.addSample( [ 0, 100 ], [0] )
    dataset.addSample( [ 0, 100 ], [1] )
    dataset.addSample( [ 0, 100 ], [0] )
    dataset.addSample( [ 0, 100 ], [1] );dataset.newSequence()
#    dataset.addSample( [ 1, 0 ], [1] );dataset.newSequence()
#    dataset.addSample( [ 1, 1 ], [0] );dataset.newSequence()
    dataset.addSample( [ 0, 0 ], [0] );dataset.newSequence()
    dataset.addSample( [ 0, 1 ], [1] );dataset.newSequence()
    dataset.addSample( [ 1, 0 ], [1] );dataset.newSequence()
    dataset.addSample( [ 1, 1 ], [0] );dataset.newSequence()
    dataset.addSample( [ 1, 1 ], [0] );
    dataset.addSample( [ 1, 1 ], [0] );
    return dataset






if __name__ == "__main__":
    main()






    #    dataset._convertToClassNb()
#        dataset.reset()
#        outs    = []
#        targets = []
#        for seq in dataset._provideSequences():
#            net.reset()
#            print
#            for inp, tar, importance in seq:
#                out = net.activate(inp)
#                print out , " ------- " , tar
#    #            print out
#    #        out = zeros(net.outdim)
#    #        out[argmax(res)]=1
#            out = argmax(out)
#            tar = argmax(tar)
#    #        print out
#            outs.append(out)
#            targets.append(tar)
#        outs = array(outs)
#        targets = array(targets)

#        print
#        print outs
#        print
#        print targets
#        print
#        print Validator.classificationPerformance(outs, targets)




