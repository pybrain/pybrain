# $Id$
# Train a mixture of Gaussians to approximate a multi-mode dataset.
# It seems fairly easy to fall into some local minimum. Good solutions 
# have errors around -200.
# This example reproduces Fig. 5.21 from Bishop (2006).
__author__ = 'Martin Felder'

from pylab import hold, scatter, subplot, title, contourf, plot, xlim, ylim, show #@UnresolvedImport
import numpy as np
from pybrain.structure.modules import LinearLayer, BiasUnit, SigmoidLayer
from pybrain.structure import FullConnection, FeedForwardNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers.mixturedensity import RPropMinusTrainerMix
from pybrain.structure.modules.mixturedensity import MixtureDensityLayer


def multigaussian(x, mean, stddev):
    """ return value of uncorrelated Gaussians at given scalar point 
    x: scalar, mean: vector, stddev: vector """
    tmp = -0.5 * ((x-mean)/stddev)**2
    return np.exp(tmp) / (np.sqrt(2.*np.pi) * stddev)


if __name__ == '__main__':    
    # build a network
    n = FeedForwardNetwork()
    # linear input layer
    n.addInputModule(LinearLayer(1, name='in'))
    # output layer of type 'outclass'
    n.addOutputModule(MixtureDensityLayer(dim=1, name='out', mix=3))
    # add bias module and connection to out module
    n.addModule(BiasUnit(name = 'bias'))
    n.addConnection(FullConnection(n['bias'], n['out']))

    # arbitrary number of hidden layers of type 'hiddenclass'
    n.addModule(SigmoidLayer(5, name='hidden'))
    n.addConnection(FullConnection(n['bias'], n['hidden']))
    
    # network with hidden layer(s), connections from in to first hidden and last hidden to out
    n.addConnection(FullConnection(n['in'], n['hidden']))
    n.addConnection(FullConnection(n['hidden'], n['out']))   
    n.sortModules()
    n._setParameters(np.random.uniform(-0.1,0.1, size=n.paramdim))
    
    # build some data
    y = np.arange(0.0, 1.0, 0.005).reshape(200,1)
    x = y + 0.3*np.sin(2*np.pi*y) + np.random.uniform(-0.1,0.1,y.size).reshape(y.size,1)
    dataset = SupervisedDataSet(1,1)
    dataset.setField('input',x)
    dataset.setField('target',y)
    
    # train the network
    #trainer = BackpropTrainerMix(n,dataset=dataset,learningrate=0.001, momentum=0.0, weightdecay=0.0001, batchlearning=False, verbose=True)
    trainer = RPropMinusTrainerMix(n,dataset=dataset,verbose=True, weightdecay=0.05)
    trainer.trainEpochs(200)
   
    # plot the density and other stuff
    subplot(223)
    dens = []
    #newx = np.arange(x.min(), x.max(), 0.01)
    newx = np.arange(0.0, 1.0, 0.01)
    newx = newx.reshape(newx.size,1)
    dataset.setField('input', newx)
    out = n.activateOnDataset(dataset)
    for pars in out:
        line = multigaussian(newx, pars[6:9], pars[3:6])
        dens.append(line[:,0]*pars[0] +line[:,1]*pars[1] +line[:,2]*pars[2]) 
        
    newx = newx.flatten()
    dens = np.array(dens).transpose()
    contourf(newx,newx,dens,30)
    title("cond. probab. dens.")
    
    subplot(221)
    out = np.array(out)
    plot(newx,out[:,0:3])
    title("mixing coefficient")

    subplot(222)
    plot(newx,out[:,6:9])
    title("means of Gaussians")

    subplot(224)
    scatter(x.flatten(),y.flatten(),marker='o',edgecolor='g',facecolors='none')
    hold(True)
    cmode = dens.argmax(axis=0)
    plot(newx,newx[cmode],"or",markersize=3)
    xlim(0,1)
    ylim(0,1)
    title("data and cond. mode")
    show()
    