__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.rl.learners.directsearch.directsearch import DirectSearchLearner
from pybrain.rl.learners.learner import DataSetLearner, ExploringLearner
from pybrain.utilities import abstractMethod
from pybrain.auxiliary import GradientDescent
from pybrain.rl.explorers.continuous.normal import NormalExplorer
from pybrain.datasets.dataset import DataSet
from pybrain.structure.networks import FeedForwardNetwork
from pybrain.structure.connections import IdentityConnection


class LoglhDataSet(DataSet):
    def __init__(self, dim):
        DataSet.__init__(self)
        self.addField('loglh', dim)
        self.linkFields(['loglh'])
        self.index = 0


class PolicyGradientLearner(DirectSearchLearner, DataSetLearner, ExploringLearner):
    """ PolicyGradientLearner is a super class for all continuous direct search
        algorithms that use the log likelihood of the executed action to update
        the weights. Subclasses are ENAC, GPOMDP, or REINFORCE.
    """

    _module = None

    def __init__(self):
        # gradient descender
        self.gd = GradientDescent()

        # create default explorer
        self._explorer = None

        # loglh dataset
        self.loglh = None

        # network to tie module and explorer together
        self.network = None


    def _setLearningRate(self, alpha):
        """ pass the alpha value through to the gradient descent object """
        self.gd.alpha = alpha

    def _getLearningRate(self):
        return self.gd.alpha

    learningRate = property(_getLearningRate, _setLearningRate)

    def _setModule(self, module):
        """ initialize gradient descender with module parameters and
            the loglh dataset with the outdim of the module. """
        self._module = module

        # initialize explorer
        self._explorer = NormalExplorer(module.outdim)

        # build network
        self._initializeNetwork()

    def _getModule(self):
        return self._module

    module = property(_getModule, _setModule)

    def _setExplorer(self, explorer):
        """ assign non-standard explorer to the policy gradient learner.
            requires the module to be set beforehand.
        """
        assert self._module

        self._explorer = explorer

        # build network
        self._initializeNetwork()

    def _getExplorer(self):
        return self._explorer

    explorer = property(_getExplorer, _setExplorer)


    def _initializeNetwork(self):
        """ build the combined network consisting of the module and
            the explorer and initializing the log likelihoods dataset.
        """
        self.network = FeedForwardNetwork()
        self.network.addInputModule(self._module)
        self.network.addOutputModule(self._explorer)
        self.network.addConnection(IdentityConnection(self._module, self._explorer))
        self.network.sortModules()

        # initialize gradient descender
        self.gd.init(self.network.params)

        # initialize loglh dataset
        self.loglh = LoglhDataSet(self.network.paramdim)


    def learn(self):
        """ calls the gradient calculation function and executes a step in direction
            of the gradient, scaled with a small learning rate alpha. """
        assert self.dataset != None
        assert self.module != None

        # calculate the gradient with the specific function from subclass
        gradient = self.calculateGradient()

        # scale gradient if it has too large values
        if max(gradient) > 1000:
            gradient = gradient / max(gradient) * 1000

        # update the parameters of the module
        p = self.gd(gradient.flatten())
        self.network._setParameters(p)
        self.network.reset()

    def explore(self, state, action):
        # forward pass of exploration
        explorative = ExploringLearner.explore(self, state, action)

        # backward pass through network and store derivs
        self.network.backward()
        self.loglh.appendLinked(self.network.derivs.copy())

        return explorative

    def reset(self):
        self.loglh.clear()

    def calculateGradient(self):
        abstractMethod()
