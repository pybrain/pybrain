# $Id$
__author__ = 'Martin Felder'

from scipy import dot, sqrt

from pybrain.supervised.trainers import BackpropTrainer


class RPropMinusTrainer(BackpropTrainer):
    """ Train the parameters of a module according to a supervised dataset (possibly sequential)
        by RProp without weight backtracking (aka RProp-, cf. [Igel&Huesken, Neurocomputing 50, 2003]) 
        and without ponderation, ie. all training samples have the same weight. """
        
    def __init__(self, module, etaminus = 0.5, etaplus = 1.2, deltamin = 1.0e-6, deltamax = 5.0, delta0 = 0.1, **kwargs):
        """ Set up training algorithm parameters, and objects associated with the trainer.
            @param module: the module whose parameters should be trained. 
            @param etaminus: factor by which step width is decreased when overstepping (0.5)
            @param etaplus: factor by which step width is increased when following gradient (1.2)
            @param delta: step width for each weight 
            @param deltamin: minimum step width (1e-6)
            @param deltamax: maximum step width (5.0)
            @param delta0: initial step width (0.1)           
        """
        BackpropTrainer.__init__(self, module, **kwargs)
        self.epoch = 0
        # set descender to RPROP mode and update parameters
        self.descent.rprop = True
        self.descent.etaplus = etaplus
        self.descent.etaminus = etaminus
        self.descent.deltamin = deltamin
        self.descent.deltamax = deltamax
        self.descent.deltanull = delta0
        self.descent.init(module.params)  # reinitialize, since mode changed

    def train(self):
        """ Train the network for one epoch """
        self.module.resetDerivatives()
        error = 0
        for seq in self.ds._provideSequences():
            e, dummy = self._calcDerivs(seq)
            error += e
        if self.verbose:
            print "epoch %6d  total error %12.5g   avg weight  %12.5g" % (self.epoch, error, sqrt((self.module.params**2).mean()))
        self.module._setParameters(self.descent(self.module.derivs - self.weightdecay*self.module.params))
        self.epoch += 1
        self.totalepochs += 1

     
    def _calcDerivs(self, seq):
        # TODO: this does not work with ImportanceDataSets! FIX by synchronizing with BackPropTrainer.
        self.module.reset()        
        for sample in seq:
            self.module.activate(sample[0])
        error = 0
        for time, sample in reversed(list(enumerate(seq))):
            _, target = sample
            outerror = target - self.module.outputbuffer[time]
            self.module.backActivate(outerror)
            error += 0.5 * dot(outerror, outerror)
        return error, 1.0
