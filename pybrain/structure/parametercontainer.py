__author__ = 'Daan Wierstra and Tom Schaul'

from scipy import size, zeros, ndarray, array
from numpy.random import randn
from numpy.random import seed

from pybrain.structure.evolvables.evolvable import Evolvable


class ParameterContainer(Evolvable):
    """ A common interface implemented by all classes which
    contains data that can change during execution (i.e. trainable parameters)
    and should be losslessly storable and retrievable to files.  """

    # standard deviation for random values, and for mutation
    stdParams = 1.
    mutationStd = 0.1

    # if this variable is set, then only the owner can set the params or the derivs of the container
    owner = None

    # a flag that enables storage of derivatives
    hasDerivatives = False

    def __init__(self, paramdim = 0, use_random_seed=True, **args):
        """ initialize all parameters with random values, normally distributed around 0

            :key stdParams: standard deviation of the values (default: 1).
            :key use_random_seed: flag to use random seed or set the seed based on connection dimensions
        """
        self.setArgs(**args)
        self.paramdim = paramdim
        self.use_random_seed = use_random_seed
        if paramdim > 0:
            self._params = zeros(self.paramdim)
            # enable derivatives if it is a instance of Module or Connection
            # CHECKME: the import can not be global?
            from pybrain.structure.modules.module import Module
            from pybrain.structure.connections.connection import Connection
            if isinstance(self, Module) or isinstance(self, Connection):
                self.hasDerivatives = True
            if self.hasDerivatives:
                self._derivs = zeros(self.paramdim)
            self.randomize()
        else:
            # We need to initialize, otherwise, forward propagation fails
            # File "/usr/local/lib/python2.7/site-packages/pybrain/supervised/trainers/rprop.py", line 42, in train
            #   e, p = self._calcDerivs(seq)
            # File "/usr/local/lib/python2.7/site-packages/pybrain/supervised/trainers/backprop.py", line 83, in _calcDerivs
            #   self.module.activate(sample[0])
            # File "/usr/local/lib/python2.7/site-packages/pybrain/structure/networks/feedforward.py", line 20, in activate
            #    return super(FeedForwardNetworkComponent, self).activate(inpt)
            #  File "/usr/local/lib/python2.7/site-packages/pybrain/structure/modules/module.py", line 106, in activate
            #    self.forward()
            #  File "/usr/local/lib/python2.7/site-packages/pybrain/structure/modules/module.py", line 73, in forward
            #    self.outputbuffer[self.offset])
            #  File "/usr/local/lib/python2.7/site-packages/pybrain/structure/networks/feedforward.py", line 33, in _forwardImplementatio
            #    c.forward()
            #  File "/usr/local/lib/python2.7/site-packages/pybrain/structure/connections/connection.py", line 77, in forward
            #    self.outmod.inputbuffer[outmodOffset, self.outSliceFrom:self.outSliceTo])
            #  File "/usr/local/lib/python2.7/site-packages/pybrain/structure/connections/full.py", line 19, in _forwardImplementation
            #    outbuf += dot(reshape(self.params, (self.outdim, self.indim)), inbuf)
            #  File "/usr/local/lib/python2.7/site-packages/pybrain/structure/parametercontainer.py", line 46, in params
            #    return self._params
            # AttributeError: 'FullConnection' object has no attribute '_params'

            self._params = zeros(paramdim)
            self._derivs = zeros(paramdim)

    @property
    def params(self):
        """ @rtype: an array of numbers. """
        return self._params

    def __len__(self):
        return self.paramdim

    def _setParameters(self, p, owner = None):
        """ :key p: an array of numbers """
        if isinstance(p, list):
            p = array(p)
        assert isinstance(p, ndarray)

        if self.owner == self:
            # the object owns it parameter array, which means it cannot be set,
            # only updated with new values.
            self._params[:] = p
        elif self.owner != owner:
            raise Exception("Parameter ownership mismatch: cannot set to new array.")
        else:
            self._params = p
            self.paramdim = size(self.params)

    @property
    def derivs(self):
        """ :rtype: an array of numbers. """
        return self._derivs

    def _setDerivatives(self, d, owner = None):
        """ :key d: an array of numbers of self.paramdim """
        assert self.owner == owner
        assert size(d) == self.paramdim
        self._derivs = d

    def resetDerivatives(self):
        """ :note: this method only sets the values to zero, it does not initialize the array. """
        assert self.hasDerivatives
        self._derivs *= 0

    def randomize(self):
        if not self.use_random_seed:
            seed(self.paramdim)
        self._params[:] = randn(self.paramdim)*self.stdParams
        if self.hasDerivatives:
            self.resetDerivatives()

    def mutate(self):
        self._params += randn(self.paramdim)*self.mutationStd
