# $Id$
# Train a mixture of Gaussians to approximate a multi-mode dataset.
# It seems fairly easy to fall into some local minimum. Good solutions 
# have errors around -200.
# This example reproduces Fig. 5.21 from Bishop (2006).
__author__ = 'Martin Felder'

import pylab as p
import numpy as np
from pybrain.structure.modules import LinearLayer, BiasUnit, SigmoidLayer
from pybrain.structure import FullConnection, FeedForwardNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers.mixturedensity import RPropMinusTrainerMix, BackpropTrainerMix
from pybrain.structure.modules.mixturedensity import MixtureDensityLayer


def multigaussian(x, mean, stddev):
    """Returns value of uncorrelated Gaussians at given scalar point.
    x: scalar
    mean: vector
    stddev: vector
    """
    tmp = -0.5 * ((x-mean)/stddev)**2
    return np.exp(tmp) / (np.sqrt(2.*np.pi) * stddev)


if __name__ == '__main__':    
    # build a network
    n = FeedForwardNetwork()
    # linear input layer
    n.addInputModule(LinearLayer(1, name='in'))
    # output layer of type 'outclass'
    N_GAUSSIANS = 3
    n.addOutputModule(MixtureDensityLayer(dim=1, name='out', mix=N_GAUSSIANS))
    # add bias module and connection to out module
    n.addModule(BiasUnit(name = 'bias'))
    n.addConnection(FullConnection(n['bias'], n['out']))

    # arbitrary number of hidden layers of type 'hiddenclass'
    n.addModule(SigmoidLayer(5, name='hidden'))
    n.addConnection(FullConnection(n['bias'], n['hidden']))
    
    # network with hidden layer(s), connections 
    # from in to first hidden and last hidden to out
    n.addConnection(FullConnection(n['in'], n['hidden']))
    n.addConnection(FullConnection(n['hidden'], n['out']))   
    n.sortModules()
    n._setParameters(np.random.uniform(-0.1, 0.1, size=n.paramdim))
    
    # build some data
    y = np.arange(0.0, 1.0, 0.005).reshape(200,1)
    x = (
        y + 
        0.3 * np.sin(2 * np.pi * y) + 
        np.random.uniform(-0.1, 0.1, y.size).reshape(y.size, 1)
    )
    dataset = SupervisedDataSet(1, 1)
    dataset.setField('input', x)
    dataset.setField('target', y)
    
    # train the network
    trainer = RPropMinusTrainerMix(n, dataset=dataset, verbose=True, 
                                   weightdecay=0.05)
    trainer.trainEpochs(200)
   
    # plot the density and other stuff
    p.subplot(2, 2, 3)
    dens = []
    newx = np.arange(0.0, 1.0, 0.01)
    newx = newx.reshape(newx.size, 1)
    dataset.setField('input', newx)
    out = n.activateOnDataset(dataset)
    for pars in out:
        stds = pars[N_GAUSSIANS:N_GAUSSIANS*2]
        means = pars[N_GAUSSIANS*2:N_GAUSSIANS*3]
        line = multigaussian(newx, means, stds)
        density = line[:,0] * pars[0]
        for gaussian in range(1, N_GAUSSIANS):
            density += line[:, gaussian] * pars[gaussian]
        dens.append(density)
        
    newx = newx.flatten()
    dens = np.array(dens).transpose()
    p.contourf(newx, newx, dens, 30)
    p.title("cond. probab. dens.")
    
    p.subplot(221)
    out = np.array(out)
    p.plot(newx, out[:,0:3])
    p.title("mixing coefficient")

    p.subplot(222)
    p.plot(newx, out[:,6:9])
    p.title("means of Gaussians")

    p.subplot(224)
    p.scatter(x.flatten(), y.flatten(), 
              marker='o', edgecolor='g', facecolors='none')
    p.hold(True)
    cmode = dens.argmax(axis=0)
    p.plot(newx, newx[cmode], "or", markersize=3)
    p.xlim(0, 1)
    p.ylim(0, 1)
    p.title("data and cond. mode")
    p.show()
    
