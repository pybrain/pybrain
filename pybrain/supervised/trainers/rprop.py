# $Id$
__author__ = 'Martin Felder'

from scipy import sqrt

from pybrain.supervised.trainers import BackpropTrainer


class RPropMinusTrainer(BackpropTrainer):
    """ Train the parameters of a module according to a supervised dataset (possibly sequential)
        by RProp without weight backtracking (aka RProp-, cf. [Igel&Huesken, Neurocomputing 50, 2003])
        and without ponderation, ie. all training samples have the same weight. """

    def __init__(self, module, etaminus=0.5, etaplus=1.2, deltamin=1.0e-6, deltamax=5.0, delta0=0.1, **kwargs):
        """ Set up training algorithm parameters, and objects associated with the trainer.

            :arg module: the module whose parameters should be trained.
            :key etaminus: factor by which step width is decreased when overstepping (0.5)
            :key etaplus: factor by which step width is increased when following gradient (1.2)
            :key delta: step width for each weight
            :key deltamin: minimum step width (1e-6)
            :key deltamax: maximum step width (5.0)
            :key delta0: initial step width (0.1)
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
        errors = 0
        ponderation = 0
        for seq in self.ds._provideSequences():
            e, p = self._calcDerivs(seq)
            errors += e
            ponderation += p
        if self.verbose:
            print("epoch {epoch:6d}  total error {error:12.5g}   avg weight  {weight:12.5g}".format(
                epoch=self.epoch,
                error=errors / ponderation,
                weight=sqrt((self.module.params ** 2).mean())))
        self.module._setParameters(self.descent(self.module.derivs - self.weightdecay * self.module.params))
        self.epoch += 1
        self.totalepochs += 1
        return errors / ponderation


