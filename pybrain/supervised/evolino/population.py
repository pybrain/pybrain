__author__ = 'Michael Isik'

from pybrain.supervised.evolino.gpopulation import Population, SimplePopulation
from pybrain.supervised.evolino.gfilter import Randomization
from pybrain.supervised.evolino.individual import EvolinoIndividual, EvolinoSubIndividual

from pybrain.tools.kwargsprocessor import KWArgsProcessor

from copy   import copy
from random import randrange


class EvolinoPopulation(Population):
    """ Evolino's population class.

        EvolinoIndividuals aren't stored directly, but there is a list of
        subpopulations.
        These subpopulations are used to generate EvolinoIndividuals on demand.

        On initialization, a prototype individual must be supplied. Its genome
        should be a list of chromosomes. A chromosome should be a list of floats.

        A subpopulation of size subPopulationSize is created for each of these
        chromosomes.

        :key nCombinations: Denotes the number of times each subindividual should
                              be built into an individual. default=1
        :key valueInitializer:
    """
    def __init__(self, individual, subPopulationSize, nCombinations=1, valueInitializer=Randomization(-0.1, 0.1), **kwargs):
        """ :key individual: A prototype individual which is used to determine
                               the structure of the genome.
            :key subPopulationSize: integer describing the size of the subpopulations
        """
        Population.__init__(self)

        self._subPopulations = []

        self.nCombinations = nCombinations

        ap = KWArgsProcessor(self, kwargs)

        ap.add('verbosity', default=0)



        genome = individual.getGenome()
        for chromosome in genome:
            self._subPopulations.append(
                EvolinoSubPopulation(chromosome, subPopulationSize, valueInitializer))

    def getIndividuals(self):
        """ Returns a set of individuals of type EvolinoIndividual. The individuals
            are generated on the fly. Note that each subpopulation has the same size.
            So the number of resulting EvolinoIndividuals is subPopulationSize,
            since each chromosome of each subpopulation will be assembled once.

            The subpopulation container is a sequence with strict order. This
            sequence is iterated subPopulationSize times. In each iteration
            one random EvolinoSubIndividual is taken from each sub population.
            After each iteration the resulting sequence of sub individuals
            is supplied to the constructor of a new EvolinoIndividual.
            All EvolinoIndividuals are collected in a set, which is finally returned.
        """
        assert len(self._subPopulations)

        individuals = set()

        for _ in range(self.nCombinations):
            subIndividualsList = [ list(sp.getIndividuals()) for sp in self._subPopulations ]

            nIndividuals = len(subIndividualsList[0])

            for _ in range(nIndividuals):
                subIndividualCombination = []
                for subIndividuals in subIndividualsList:
                    sub_individual = subIndividuals.pop(randrange(len(subIndividuals)))
                    subIndividualCombination.append(sub_individual)
                individuals.add(EvolinoIndividual(subIndividualCombination))

        return individuals

    def getSubPopulations(self):
        """ Returns a shallow copy of the list of subpopulation. """
        return copy(self._subPopulations)



    def setIndividualFitness(self, individual, fitness):
        """ The fitness value is not stored directly inside this population,
            but is propagated to the subpopulations of all the subindividuals
            of which the individual consists of.
            The individual's fitness value is only adjusted if its bigger than
            the old value.
            To reset these values use clearFitness().
        """


        # additive fitness distribution
#        subIndividuals = individual.getSubIndividuals()
#        for i,sp in enumerate(self._subPopulations):
#            sp.addIndividualFitness( subIndividuals[i], fitness )

        # max fitness distribution
        subIndividuals = individual.getSubIndividuals()
        for i, sp in enumerate(self._subPopulations):
            sub_individual = subIndividuals[i]
            old_fitness = sp.getIndividualFitness(sub_individual)
            if old_fitness < fitness:
                sp.setIndividualFitness(sub_individual, fitness)



    def clearFitness(self):
        """ Clears all fitness values of all subpopulations. """
        for sp in self._subPopulations:
            sp.clearFitness()




class EvolinoSubPopulation(SimplePopulation):
    """ The class for Evolino subpopulations. Mostly the same as SimplePopulation
        but with a few extensions.
        It contains a set of EvolinoSubIndividuals.

        On initialization, a prototype individual is created from the prototype
        chromosome. This individual is then cloned and added so that the
        population exists of maxNIndividuals individuals.

        The genomes of these clones are then randomized by the Randomization
        operator.
    """
    def __init__(self, chromosome, maxNIndividuals, valueInitializer=Randomization(-0.1, 0.1), **kwargs):
        """ :key chromosome: The prototype chromosome
            :key maxNIndividuals: The maximum allowed number of individuals
        """
        SimplePopulation.__init__(self)

        self._prototype = EvolinoSubIndividual(chromosome)

        self._maxNIndividuals = maxNIndividuals
        self._valueInitializer = valueInitializer


        self.setArgs(**kwargs)


        for _ in range(maxNIndividuals):
            self.addIndividual(self._prototype.copy())
        self._valueInitializer.apply(self)


    def setArgs(self, **kwargs):
        for key, val in kwargs.iteritems():
            getattr(self, key)
            setattr(self, key, val)


    def getMaxNIndividuals(self):
        """ Returns the maximum allowed number of individuals """
        return self._maxNIndividuals

    def addIndividualFitness(self, individual, fitness):
        """ Add fitness to the individual's fitness value.
            :key fitness: a float value denoting the fitness
        """
        self._fitness[individual] += fitness



