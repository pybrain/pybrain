"""
    >>> from pybrain.datasets.supervised import SupervisedDataSet
    >>> from pybrain.supervised.trainers import BackpropTrainer
    >>> from pybrain import FeedForwardNetwork
    >>> from pybrain.structure import LinearLayer, SigmoidLayer, FullConnection
    >>> from random import randrange
    >>> dataset = SupervisedDataSet(6, 2)
    >>> for i in range(1000):
    ...     state = [randrange(0, 15), 
    ...              randrange(-70, 50), 
    ...              randrange(-70, 50), 
    ...              randrange(-70, 50), 
    ...              randrange(-70, 50), 
    ...              float(randrange(1, 5))/20.]
    ...     action = [float(randrange(-1, 1))/10.0, 
    ...               randrange(0, 1)]
    ...     dataset.addSample(state, action)
    >>> 
    >>> net = FeedForwardNetwork()
    >>> 
    >>> net.addInputModule(LinearLayer(6, name='in'))
    >>> net.addModule(SigmoidLayer(40, name='hidden_0'))
    >>> net.addModule(SigmoidLayer(16, name='hidden_1'))
    >>> net.addOutputModule(LinearLayer(2, name='out'))
    >>> 
    >>> net.addConnection(FullConnection(net['in'], net['hidden_0']))
    >>> net.addConnection(FullConnection(net['hidden_0'], net['hidden_1']))
    >>> net.addConnection(FullConnection(net['hidden_1'], net['out']))
    >>> 
    >>> net.sortModules()
    >>> 
    >>> trainer = BackpropTrainer(net,
    ...                           dataset=dataset,
    ...                           learningrate=0.01,
    ...                           lrdecay=1,
    ...                           momentum=0.5,
    ...                           verbose=False,
    ...                           weightdecay=0,
    ...                           batchlearning=False)
    >>> 
    >>> trainingErrors, validationErrors = trainer.trainUntilConvergence(
    ...    dataset=dataset, 
    ...    maxEpochs=10)
"""



__author__ = 'Steffen Kampmann, steffen.kampmann@gmail.com'

from pybrain.tests import runModuleTestSuite

if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))
