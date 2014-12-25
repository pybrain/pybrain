from __future__ import print_function

#!/usr/bin/env python
""" A simple recurrent neural network that detects parity for arbitrary sequences. """

__author__ = 'Tom Schaul (tom@idsia.ch)'

from datasets import ParityDataSet #@UnresolvedImport
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.structure import RecurrentNetwork, LinearLayer, TanhLayer, BiasUnit, FullConnection


def buildParityNet():
    net = RecurrentNetwork()
    net.addInputModule(LinearLayer(1, name = 'i'))
    net.addModule(TanhLayer(2, name = 'h'))
    net.addModule(BiasUnit('bias'))
    net.addOutputModule(TanhLayer(1, name = 'o'))
    net.addConnection(FullConnection(net['i'], net['h']))
    net.addConnection(FullConnection(net['bias'], net['h']))
    net.addConnection(FullConnection(net['bias'], net['o']))
    net.addConnection(FullConnection(net['h'], net['o']))
    net.addRecurrentConnection(FullConnection(net['o'], net['h']))
    net.sortModules()

    p = net.params
    p[:] = [-0.5, -1.5, 1, 1, -1, 1, 1, -1, 1]
    p *= 10.

    return net

def evalRnnOnSeqDataset(net, DS, verbose = False, silent = False):
    """ evaluate the network on all the sequences of a dataset. """
    r = 0.
    samples = 0.
    for seq in DS:
        net.reset()
        for i, t in seq:
            res = net.activate(i)
            if verbose:
                print(t, res)
            r += sum((t-res)**2)
            samples += 1
        if verbose:
            print('-'*20)
    r /= samples
    if not silent:
        print('MSE:', r)
    return r

if __name__ == "__main__":
    N = buildParityNet()
    DS = ParityDataSet()
    evalRnnOnSeqDataset(N, DS, verbose = True)
    print('(preset weights)')
    N.randomize()
    evalRnnOnSeqDataset(N, DS)
    print('(random weights)')


    # Backprop improves the network performance, and sometimes even finds the global optimum.
    N.reset()
    bp = BackpropTrainer(N, DS, verbose = True)
    bp.trainEpochs(5000)
    evalRnnOnSeqDataset(N, DS)
    print('(backprop-trained weights)')
