__author__ = 'Michael Isik'


from numpy import Infinity

from pybrain.supervised.trainers.trainer import Trainer
from pybrain.supervised.evolino.population import EvolinoPopulation
from pybrain.supervised.evolino.individual import EvolinoSubIndividual
from pybrain.supervised.evolino.filter import EvolinoEvaluation, EvolinoSelection, EvolinoReproduction, EvolinoBurstMutation
from pybrain.supervised.evolino.gfilter import Randomization
from pybrain.supervised.evolino.variate import CauchyVariate
from pybrain.tools.kwargsprocessor import KWArgsProcessor


class EvolinoTrainer(Trainer):
    """ The Evolino trainer class.

        Use a network as module that should be trained. There are some restrictions
        the network must follow. Basically, it should be a simple lstm network.
        For more details on these restrictions read NetworkWrapper's documentaion.
    """
    initialWeightRange = property(lambda self: self._initialWeightRange)
    subPopulationSize = property(lambda self: self._subPopulationSize)
    nCombinations = property(lambda self: self._nCombinations)
    nParents = property(lambda self: self._nParents)
    initialWeightRange = property(lambda self: self._initialWeightRange)
    mutationAlpha = property(lambda self: self._mutationAlpha)
    mutationVariate = property(lambda self: self._mutationVariate)
    wtRatio = property(lambda self: self._wtRatio)
    weightInitializer = property(lambda self: self._weightInitializer)
#    burstMutation        = property(lambda self: self._burstMutation)
    backprojectionFactor = property(lambda self: self._backprojectionFactor)

    def __init__(self, evolino_network, dataset, **kwargs):
        """
            :key subPopulationSize: Size of the subpopulations.
            :key nCombinations: Number of times each chromosome is built into an individual. default=1
            :key nParents: Number of individuals left in a subpopulation after selection.
            :key initialWeightRange: Range of the weights of the RNN after initialization. default=(-0.1,0.1)
            :key weightInitializer: Initializer object for the weights of the RNN. default=Randomization(...)
            :key mutationAlpha: The mutation's intensity. default=0.01
            :key mutationVariate: The variate used for mutation. default=CauchyVariate(...)
            :key wtRatio: The quotient: washout-time/training-time. Needed to
                            split the sequences into washout phase and training phase.
            :key nBurstMutationEpochs: Number of epochs without increase of fitness in a row,
                                         before burstmutation is applied. default=Infinity
            :key backprojectionFactor: Weight of the backprojection. Usually
                                         supplied through evolino_network.
            :key selection: Selection object for evolino
            :key reproduction: Reproduction object for evolino
            :key burstMutation: BurstMutation object for evolino
            :key evaluation: Evaluation object for evolino
            :key verbosity: verbosity level
        """
        Trainer.__init__(self, evolino_network)

        self.network = evolino_network
        self.setData(dataset)

        ap = KWArgsProcessor(self, kwargs)

        # misc
        ap.add('verbosity', default=0)

        # population
        ap.add('subPopulationSize', private=True, default=8)
        ap.add('nCombinations', private=True, default=4)
        ap.add('nParents', private=True, default=None)
        ap.add('initialWeightRange', private=True, default=(-0.1, 0.1))
        ap.add('weightInitializer', private=True, default=Randomization(self._initialWeightRange[0], self._initialWeightRange[1]))

        # mutation
        ap.add('mutationAlpha', private=True, default=0.01)
        ap.add('mutationVariate', private=True, default=CauchyVariate(0, self._mutationAlpha))

        # evaluation
        ap.add('wtRatio', private=True, default=(1, 3))

        # burst mutation
        ap.add('nBurstMutationEpochs', default=Infinity)

        # network
        ap.add('backprojectionFactor', private=True, default=float(evolino_network.backprojectionFactor))
        evolino_network.backprojectionFactor = self._backprojectionFactor

        # aggregated objects
        ap.add('selection', default=EvolinoSelection())
        ap.add('reproduction', default=EvolinoReproduction(mutationVariate=self.mutationVariate))
        ap.add('burstMutation', default=EvolinoBurstMutation())
        ap.add('evaluation', default=EvolinoEvaluation(evolino_network, self.ds, **kwargs))

        self.selection.nParents = self.nParents

        self._population = EvolinoPopulation(
            EvolinoSubIndividual(evolino_network.getGenome()),
            self._subPopulationSize,
            self._nCombinations,
            self._weightInitializer
            )

        filters = []
        filters.append(self.evaluation)
        filters.append(self.selection)
        filters.append(self.reproduction)

        self._filters = filters

        self.totalepochs = 0
        self._max_fitness = self.evaluation.max_fitness
        self._max_fitness_epoch = self.totalepochs

    def setDataset(self, dataset):
        self.evaluation.dataset = dataset

    def trainOnDataset(self, *args, **kwargs):
        """ Not implemented """
        raise NotImplementedError()

    def train(self):
        """ Evolve for one epoch. """
        self.totalepochs += 1

        if self.totalepochs - self._max_fitness_epoch >= self.nBurstMutationEpochs:
            if self.verbosity: print("RUNNING BURST MUTATION")
            self.burstMutate()
            self._max_fitness_epoch = self.totalepochs


        for filter in self._filters:
            filter.apply(self._population)

        if self._max_fitness < self.evaluation.max_fitness:
            if self.verbosity: print("GAINED FITNESS: ", self._max_fitness, " -->" , self.evaluation.max_fitness, "\n")
            self._max_fitness = self.evaluation.max_fitness
            self._max_fitness_epoch = self.totalepochs
        else:
            if self.verbosity: print("DIDN'T GAIN FITNESS:", "best =", self._max_fitness, "    current-best = ", self.evaluation.max_fitness, "\n")

    def burstMutate(self):
        self.burstMutation.apply(self._population)

