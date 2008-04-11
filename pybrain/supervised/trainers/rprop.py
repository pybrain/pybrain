# $Id$
__author__ = 'Martin Felder'

from scipy import zeros, dot, ones, argmax, sign

from pybrain.supervised.trainers import BackpropTrainer


class RPropMinusTrainer(BackpropTrainer):
    """ Train the parameters of a module according to a supervised dataset (possibly sequential)
        by RProp without weight backtracking (aka RProp-) and without ponderation. """
        
    def __init__(self, module, etaminus = 0.5, etaplus = 1.2, deltamin = 1e-6, deltamax = 5.0, delta0 = 0.1, **kwargs):
        """ @param module: the module whose parameters should be trained. 
            @param etaminus: factor by which step width is decreased when overstepping (0.5)
            @param etaplus: factor by which step width is increased when following gradient (1.2)
            @param delta: step width for each weight 
            @param deltamin: minimum step width (1e-6)
            @param deltamax: maximum step width (5.0)
            @param delta0: initial step width (0.1)           
        """
        BackpropTrainer.__init__(self, module, **kwargs)
        self.epoch = 0
        # save the individual stepwidths Delta, and the last gradient
        self.etaplus = etaplus
        self.etaminus = etaminus
        self.deltamin = deltamin
        self.deltamax = deltamax
        self.delta = zeros(module.paramdim) + delta0
        self.prevgrad = zeros(module.paramdim)    

    def train(self):
        """ Train the network for one epoch """
        self.module.resetDerivatives()
        error = 0
        ponderation = 0.
        for seq in self.ds._provideSequences():
            e = self._calcDerivs(seq)
            error += e
        if self.verbose:
            print "epoch %6d  total error %12.5g" % (self.epoch, error)
        self.updateWeights()
        self.epoch += 1
        self.totalepochs += 1

    def updateWeights(self):
        """ Update network weights and step width parameters """
        gradient = self.module.getDerivatives()
        w = self.module.getParameters() 

        # update weights
        w += sign(gradient) * self.delta
        
        # update weight steps for wm
        dirSwitch = gradient * self.prevgrad
        self.delta[dirSwitch > 0] *= self.etaplus
        idx =  dirSwitch < 0
        self.delta[idx] *= self.etaminus
        gradient[idx] = 0
        
        # limit growth and shrinkage of Deltas
        self.delta.clip(self.deltamin, self.deltamax)
        
        # save stuff for next iteration
        self.prevgrad = gradient.copy()
        
    
    def _calcDerivs(self, seq):
        self.module.reset()        
        for time, sample in enumerate(seq):
            input = sample[0]      
            self.module.inputbuffer[time] = input
            self.module.forward()
        error = 0
        for time, sample in reversed(list(enumerate(seq))):
            dummy, target = sample
            self.module.outputerror[time] = (target - self.module.outputbuffer[time])
            self.module.backward()
            error += 0.5 * dot(self.module.outputerror[time], self.module.outputerror[time])
        return error
