__author__ = 'Michael Isik'

from pybrain.supervised.evolino.gfilter import Filter, SimpleMutation
from pybrain.supervised.evolino.variate import CauchyVariate
from pybrain.supervised.evolino.population import SimplePopulation
from pybrain.tools.validation import Validator
from pybrain.tools.kwargsprocessor import KWArgsProcessor

from numpy import array, dot, concatenate, Infinity
from scipy.linalg import pinv2
from copy import deepcopy





class EvolinoEvaluation(Filter):
    """ Evaluate all individuals of the Evolino population, and store their
        fitness value inside the population.
    """

    def __init__(self, evolino_network, dataset, **kwargs):
        """ :key evolino_network: an instance of NetworkWrapper()
            :key dataset: The evaluation dataset
            :key evalfunc: Compares output to target values and returns a scalar, denoting the fitness.
                             Defaults to -mse(output, target).
            :key wtRatio: Float array of two values denoting the ratio between washout and training length.
                            Defaults to [1,2]
            :key verbosity: Verbosity level. Defaults to 0
        """
        Filter.__init__(self)
        ap = KWArgsProcessor(self, kwargs)

        ap.add('verbosity', default=0)
        ap.add('evalfunc', default=lambda output, target:-Validator.MSE(output, target))
        ap.add('wtRatio', default=array([1, 2], float))

        self.network = evolino_network
        self.dataset = dataset
        self.max_fitness = -Infinity


    def _evaluateNet(self, net, dataset, wtRatio):
        """ Evaluates the performance of net on the given dataset.
            Returns the fitness value.

            :key net: Instance of EvolinoNetwork to evaluate
            :key dataset: Sequences to test the net on
            :key wtRatio: See __init__
        """

        # === extract sequences from dataset ===
        numSequences = dataset.getNumSequences()
        washout_sequences = []
        training_sequences = []
        for i in xrange(numSequences):
            sequence = dataset.getSequence(i)[1]
            training_start = int(wtRatio * len(sequence))
            washout_sequences.append(sequence[                  : training_start   ])
            training_sequences.append(sequence[ training_start   :                  ])


        # === collect raw output (denoted by phi) ===
        phis = []
        for i in range(numSequences):
            net.reset()
            net.washout(washout_sequences[i])
            phi = net.washout(training_sequences[i])
            phis.append(phi)


        # === calculate and set weights of linear output layer ===
        PHI = concatenate(phis).T
        PHI_INV = pinv2(PHI)
        TARGET = concatenate(training_sequences).T
        W = dot(TARGET, PHI_INV)
        net.setOutputWeightMatrix(W)


        # === collect outputs by applying the newly configured network ===
        outputs = []
        for i in range(numSequences):
            out = net.extrapolate(washout_sequences[i], len(training_sequences[i]))
            outputs.append(out)


        # === calculate fitness value ===
        OUTPUT = concatenate(outputs)
        TARGET = concatenate(training_sequences)
        fitness = self.evalfunc(OUTPUT, TARGET)


        return fitness



    def apply(self, population):
        """ Evaluate each individual, and store fitness inside population.
            Also calculate and set the weight matrix W of the linear output layer.

            :arg population: Instance of EvolinoPopulation
        """
        net = self.network
        dataset = self.dataset
        population.clearFitness()
        best_W = None
        best_fitness = -Infinity


        # iterate all individuals. Note, that these individuals are created on the fly
        for individual in population.getIndividuals():

            # load the individual's genome into the weights of the net
            net.setGenome(individual.getGenome())
            fitness = self._evaluateNet(net, dataset, self.wtRatio)
            if self.verbosity > 1:
                print("Calculated fitness for individual", id(individual), " is ", fitness)

            # set the individual fitness
            population.setIndividualFitness(individual, fitness)

            if best_fitness < fitness:
                best_fitness = fitness
                best_genome = deepcopy(individual.getGenome())
                best_W = deepcopy(net.getOutputWeightMatrix())

        net.reset()
        net.setGenome(best_genome)
        net.setOutputWeightMatrix(best_W)


        # store fitness maximum to use it for triggering burst mutation
        self.max_fitness = best_fitness





class EvolinoSelection(Filter):
    """ Evolino's selection operator.
        Set its nParents attribute at any time.
        nParents specifies the number of individuals not to be deleted.
        If nParents equals None, EvolinoSubSelection will use its
        default value.
    """
    def __init__(self):
        Filter.__init__(self)
        self.nParents = None
        self.sub_selection = EvolinoSubSelection()

    def apply(self, population):
        """ The subpopulations of the EvolinoPopulation are iterated and forwarded
            to the EvolinoSubSelection() operator.

            :arg population: object of type EvolinoPopulation
        """
        self.sub_selection.nParents = self.nParents
        for sp in population.getSubPopulations():
            self.sub_selection.apply(sp)




class EvolinoReproduction(Filter):
    """ Evolino's reproduction operator """
    def __init__(self, **kwargs):
        """ :key **kwargs: will be forwarded to the EvolinoSubReproduction constructor
        """
        Filter.__init__(self)
        self._kwargs = kwargs


    def apply(self, population):
        """ The subpopulations of the EvolinoPopulation are iterated and forwarded
            to the EvolinoSubReproduction() operator.

            :arg population: object of type EvolinoPopulation
        """
        sps = population.getSubPopulations()
        reproduction = EvolinoSubReproduction(**self._kwargs)
        for sp in sps:
            reproduction.apply(sp)


class EvolinoBurstMutation(Filter):
    """ The burst mutation operator for evolino """
    def __init__(self, **kwargs):
        """ :key **kwargs: will be forwarded to the EvolinoSubReproduction constructor
        """
        Filter.__init__(self)
        self._kwargs = kwargs

    def apply(self, population):
        """ Keeps just the best fitting individual of each subpopulation.
            All other individuals are erased. After that, the kept best fitting
            individuals will be used for reproduction, in order to refill the
            sub-populations.
        """
        sps = population.getSubPopulations()
        for sp in sps:
            n_toremove = sp.getIndividualsN() - 1
            sp.removeWorstIndividuals(n_toremove)
            reproduction = EvolinoSubReproduction(**self._kwargs)
            reproduction.apply(sp)



# ==================================================== SubPopulation related ===



class EvolinoSubSelection(Filter):
    """ Selection operator for EvolinoSubPopulation objects
        Specify its nParents attribute at any time. See EvolinoSelection.
    """
    def __init__(self):
        Filter.__init__(self)

    def apply(self, population):
        """ Simply removes some individuals with lowest fitness values
        """

        n = population.getIndividualsN()
        if self.nParents is None:
            nKeep = n / 4
        else:
            nKeep = self.nParents

        assert nKeep >= 0
        assert nKeep <= n

        population.removeWorstIndividuals(n - nKeep)





class EvolinoSubReproduction(Filter):
    """ Reproduction operator for EvolinoSubPopulation objects.
    """
    def __init__(self, **kwargs):
        """ :key verbosity: Verbosity level
            :key mutationVariate: Variate used for mutation. Defaults to None
            :key mutation: Defaults to EvolinoSubMutation
        """
        Filter.__init__(self)

        ap = KWArgsProcessor(self, kwargs)
        ap.add('verbosity', default=0)
        ap.add('mutationVariate', default=None)
        ap.add('mutation', default=EvolinoSubMutation())

        if self.mutationVariate is not None:
            self.mutation.mutationVariate = self.mutationVariate




    def apply(self, population):
        """ First determines the number of individuals to be created.
            Then clones the fittest individuals (=parents), mutates these clones
            and adds them to the population.
        """
        max_n = population.getMaxNIndividuals()
        n = population.getIndividualsN()
        freespace = max_n - n

        best = population.getBestIndividualsSorted(freespace)
        children = set()
        while True:
            if len(children) >= freespace: break
            for parent in best:
                children.add(parent.copy())
                if len(children) >= freespace: break

        dummy_population = SimplePopulation()
        dummy_population.addIndividuals(children)
        self.mutation.apply(dummy_population)
        population.addIndividuals(dummy_population.getIndividuals())

        assert population.getMaxNIndividuals() == population.getIndividualsN()





class EvolinoSubMutation(SimpleMutation):
    """ Mutation operator for EvolinoSubPopulation objects.
        Like SimpleMutation, except, that CauchyVariate is used by default.
    """
    def __init__(self, **kwargs):
        SimpleMutation.__init__(self)

        ap = KWArgsProcessor(self, kwargs)
        ap.add('mutationVariate', default=CauchyVariate())
        self.mutationVariate.alpha = 0.001




