# $Id$
__author__ = 'Martin Felder'

import numpy as np
from pybrain.supervised.trainers import RPropMinusTrainer, BackpropTrainer
from pybrain.structure.modules.mixturedensity import MixtureDensityLayer

def gaussian(x, mean, stddev):
    """ return value of homogenous Gaussian at given vector point 
    x: vector, mean: vector, stddev: scalar """
    tmp = -0.5 * sum(((x-mean)/stddev)**2)
    return np.exp(tmp) / (np.power(2.*np.pi, 0.5*len(x)) * stddev)
    

class BackpropTrainerMix(BackpropTrainer):
    """ Trainer for mixture model network. See Bishop 2006, Eqn. 5.153-5.157.
    Due to PyBrain conventions it is more convenient (if not pretty) to treat the
    MixtureDensityLayer as having a linear transfer function, and calculate
    its derivative here."""
    
    def setData(self, dataset):
        # different output dimension check
        self.ds = dataset
        if dataset:
            assert dataset.indim == self.module.indim
            assert dataset.outdim == self.module.modulesSorted[-1].nDims

    def _calcDerivs(self, seq):
        """ calculate derivatives assuming we have a Network with a MixtureDensityLayer as output """
        assert isinstance(self.module.modulesSorted[-1], MixtureDensityLayer)
        
        self.module.reset()       
        for time, sample in enumerate(seq):
            input = sample[0]      
            self.module.inputbuffer[time] = input
            self.module.forward()
        error = 0
        nDims = self.module.modulesSorted[-1].nDims
        nGauss = self.module.modulesSorted[-1].nGaussians
        for time, sample in reversed(list(enumerate(seq))):

            # Should these three lines be inside this 'for' block
            # or outside?  I moved them inside - Jack
            gamma = []
            means = []
            stddevs = []

            dummy, target = sample
            par = self.module.outputbuffer[time] # parameters for mixture
            # calculate error contributions from all Gaussians in the mixture
            for k in range(nGauss):
                coeff = par[k]
                stddevs.append(par[k+nGauss])
                idxm = 2*nGauss + k*nDims
                means.append(par[idxm:idxm+nDims])
                gamma.append(coeff * gaussian(target, means[-1], stddevs[-1]))
                
            # calculate error for this pattern, and posterior for target
            sumg = sum(gamma)
            error -= np.log(sumg)
            gamma = np.array(gamma)/sumg
            
            invvariance = 1./par[nGauss:2*nGauss]**2
            invstddev = 1./np.array(stddevs)
            
            # calculate gradient wrt. mixture coefficients
            grad_c = par[0:nGauss] - gamma

            # calculate gradient wrt. standard deviations
            grad_m = []
            grad_s = []
            for k in range(nGauss):
                delta = means[k]-target
                grad_m.append(gamma[k]*delta*invvariance[k])
                grad_s.append(-gamma[k]*(np.dot(delta,delta)*invvariance[k]*invstddev[k] - invstddev[k]))
            
            self.module.outputerror[time] = -np.r_[grad_c,grad_s,np.array(grad_m).flatten()]
            self.module.backward()

        return error, 1.0
        

class RPropMinusTrainerMix(BackpropTrainerMix,RPropMinusTrainer):
    """ RProp trainer for mixture model network. See Bishop 2006, Eqn. 5.153-5.157. """
    dummy = 0  
