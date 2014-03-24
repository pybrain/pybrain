__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import zeros, randn
from random import random, sample, gauss

from pybrain.structure.evolvables.topology import TopologyEvolvable


class MaskedParameters(TopologyEvolvable):
    """ A module with a binary mask that can disable (=zero) parameters.
    If no maximum is set, the mask can potentially have all parameters enabled.
    The maxComplexity represents the number of allowed enabled parameters. """

    maskFlipProbability = 0.05
    mutationStdev = 0.1

    # number of bits in the mask that can be maximally on at once (None = all)
    # Note: there must always be at least one on
    maxComplexity = None

    # probability of mask bits being on in a random mask (subject to the constraint above)
    maskOnProbability = 0.5

    # when accessed through .params, the masked values are included (and have value zero).
    returnZeros = False

    def __init__(self, pcontainer, **args):
        TopologyEvolvable.__init__(self, pcontainer, **args)
        if self.maxComplexity == None:
            self.maxComplexity = self.pcontainer.paramdim
        self.randomize()
        self.maskableParams = self.pcontainer.params.copy()
        self._applyMask()

    def _applyMask(self):
        """ apply the mask to the module. """
        self.pcontainer._params[:] = self.mask*self.maskableParams

    @property
    def paramdim(self):
        if self.returnZeros:
            return self.pcontainer.paramdim
        else:
            return sum(self.mask)

    @property
    def params(self):
        """ returns an array with (usually) only the unmasked parameters """
        if self.returnZeros:
            return self.pcontainer.params
        else:
            x = zeros(self.paramdim)
            paramcount = 0
            for i in range(len(self.maskableParams)):
                if self.mask[i] == True:
                    x[paramcount] = self.maskableParams[i]
                    paramcount += 1
            return x

    def _setParameters(self, x):
        """ sets only the unmasked parameters """
        paramcount = 0
        for i in range(len(self.maskableParams)):
            if self.mask[i] == True:
                self.maskableParams[i] = x[paramcount]
                paramcount += 1
        self._applyMask()

    def randomize(self, **args):
        """ an initial, random mask (with random params)
        with as many parameters enabled as allowed"""
        self.mask = zeros(self.pcontainer.paramdim, dtype=bool)
        onbits = []
        for i in range(self.pcontainer.paramdim):
            if random() > self.maskOnProbability:
                self.mask[i] = True
                onbits.append(i)
        over = len(onbits) - self.maxComplexity
        if over > 0:
            for i in sample(onbits, over):
                self.mask[i] = False
        self.maskableParams = randn(self.pcontainer.paramdim)*self.stdParams
        self._applyMask()

    def topologyMutate(self):
        """ flips some bits on the mask
        (but do not exceed the maximum of enabled parameters). """
        for i in range(self.pcontainer.paramdim):
            if random() < self.maskFlipProbability:
                self.mask[i] = not self.mask[i]
        tooMany = sum(self.mask) - self.maxComplexity
        for i in range(tooMany):
            while True:
                ind = int(random()*self.pcontainer.paramdim)
                if self.mask[ind]:
                    self.mask[ind] = False
                    break
        if sum(self.mask) == 0:
            # CHECKME: minimum of one needs to be on
            ind = int(random()*self.pcontainer.paramdim)
            self.mask[ind] = True

        self._applyMask()

    def mutate(self):
        """ add some gaussian noise to all parameters."""
        # CHECKME: could this be partly outsourced to the pcontainer directly?
        for i in range(self.pcontainer.paramdim):
            self.maskableParams[i] += gauss(0, self.mutationStdev)
        self._applyMask()
