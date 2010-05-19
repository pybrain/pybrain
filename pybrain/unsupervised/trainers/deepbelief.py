# -*- coding: utf-8 -*-

__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'
__version__ = '$Id$'


import copy

from pybrain.datasets import SupervisedDataSet, UnsupervisedDataSet
from pybrain.structure import BiasUnit, FeedForwardNetwork, FullConnection
from pybrain.structure.networks.rbm import Rbm
from pybrain.structure.modules.neuronlayer import NeuronLayer
from pybrain.supervised.trainers import Trainer
from pybrain.unsupervised.trainers.rbm import (RbmBernoulliTrainer,
                                               RbmGaussTrainer)


class DeepBeliefTrainer(Trainer):
    """Trainer for deep networks.

    Trains the network by greedily training layer after layer with the
    RbmGibbsTrainer.

    The network that is being trained is assumed to be a chain of layers that
    are connected with full connections and feature a bias each.

    The behaviour of the trainer is undefined for other cases.
    """

    trainers = {
        'bernoulli': RbmBernoulliTrainer,
        'gauss': RbmGaussTrainer,
    }

    def __init__(self, net, dataset, epochs=50,
                 cfg=None, distribution='bernoulli'):
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
        self.trainerKlass = self.trainers[distribution]

    def trainRbm(self, rbm, dataset):
        trainer = self.trainerKlass(rbm, dataset, self.cfg)
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
        piecenet.addInputModule(copy.deepcopy(self.net.inmodules[0]))
        # Add a bias
        bias = BiasUnit()
        piecenet.addModule(bias)
        # Add the first visible layer
        firstRbm = self.iterRbms().next()
        visible = copy.deepcopy(firstRbm.visible)
        piecenet.addModule(visible)
        # For saving the rbms and their inverses
        self.invRbms = []
        self.rbms = []
        for rbm in self.iterRbms():
            self.net.sortModules()
            # Train the first layer with an rbm trainer for `epoch` epochs.
            trainer = self.trainerKlass(rbm, dataset, self.cfg)
            for _ in xrange(self.epochs):
                trainer.train()
            self.invRbms.append(trainer.invRbm)
            self.rbms.append(rbm)
            # Add the connections and the hidden layer of the rbm to the net.
            hidden = copy.deepcopy(rbm.hidden)
            biascon = FullConnection(bias, hidden)
            biascon.params[:] = rbm.biasWeights
            con = FullConnection(visible, hidden)
            con.params[:] = rbm.weights

            piecenet.addConnection(biascon)
            piecenet.addConnection(con)
            piecenet.addModule(hidden)
            # Overwrite old outputs
            piecenet.outmodules = [hidden]
            piecenet.outdim = rbm.hiddenDim
            piecenet.sortModules()

            dataset = UnsupervisedDataSet(rbm.hiddenDim)
            for sample, in self.dataset:
                new_sample = piecenet.activate(sample)
                dataset.addSample(new_sample)
            visible = hidden
