#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-

__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'
__version__ = '$Id$'


from pybrain.datasets import SupervisedDataSet, UnsupervisedDataSet
from pybrain.structure import BiasUnit, FeedForwardNetwork
from pybrain.structure.networks.rbm import Rbm
from pybrain.structure.modules.neuronlayer import NeuronLayer
from pybrain.supervised.trainers import Trainer
from pybrain.unsupervised.trainers.rbm import RbmGibbsTrainer


class DeepBeliefTrainer(Trainer):
    """Trainer for deep networks.
    
    Trains the network by greedily training layer after layer with the 
    RbmGibbsTrainer.
    
    The network that is being trained is assumed to be a chain of layers that
    are connected with full connections and feature a bias each. 
    
    The behaviour of the trainer is undefined for other cases.
    """
    
    def __init__(self, net, dataset, epochs=50, cfg=None):
        if isinstance(dataset, SupervisedDataSet):
            self.datasetfield = 'input'
        elif isinstance(dataset, UnsupervisedDataSet):
            self.datasetfield = 'sample'
        else:
            raise ValueError("Wrong dataset class.")
        self.net = net
        self.net.sortModules()
        self.dataset = dataset
        self.epochs = epochs
        self.cfg = cfg
        
    def trainRbm(self, rbm, dataset):
        trainer = RbmGibbsTrainer(rbm, dataset, self.cfg)
        for _ in xrange(self.epochs):
            trainer.train()
        return rbm

    def iterRbms(self):
        """Yield every two layers as an rbm."""
        layers = [i for i in self.net.modulesSorted 
                  if isinstance(i, NeuronLayer) and not isinstance(i, BiasUnit)]
        # There will be a single bias.
        bias = [i for i in self.net.modulesSorted if isinstance(i, BiasUnit)][0]
        layercons = (self.net.connections[i][0] for i in layers)
        # The biascons will not be sorted; we have to sort them to zip nicely 
        # with the corresponding layers.
        biascons = self.net.connections[bias]
        biascons.sort(key=lambda c: layers.index(c.outmod))
        modules = zip(layers, layers[1:], layercons, biascons)
        for visible, hidden, layercon, biascon in modules:
            rbm = Rbm.fromModules(visible, hidden, bias, 
                                  layercon, biascon)
            yield rbm
        
    def train(self):
        # We will build up a network piecewise in order to create a new dataset
        # for each layer.
        dataset = self.dataset
        piecenet = FeedForwardNetwork()
        piecenet.addInputModule(self.net.inmodules[0])
        bias = [i for i in self.net.modulesSorted if isinstance(i, BiasUnit)][0]
        piecenet.addModule(bias)
        for rbm in self.iterRbms():
            self.net.sortModules()
            # Train the first layer with an rbm trainer for `epoch` epochs.
            for _ in xrange(self.epochs):
                RbmGibbsTrainer(rbm, dataset).train()
            # Add the connections and the hidden layer of the rbm to the net.
            piecenet.addConnection(rbm.biascon)
            piecenet.addConnection(rbm.con)
            piecenet.addModule(rbm.hidden)
            # Overwrite old outputs
            piecenet.outmodules = [rbm.hidden]
            piecenet.outdim = rbm.hiddenDim
            piecenet.sortModules()
            
            dataset = UnsupervisedDataSet(rbm.hiddenDim)
            for sample, in self.dataset:
                new_sample = piecenet.activate(sample)
                dataset.addSample(new_sample)