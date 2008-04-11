
__author__ = 'Michael Isik'

from pybrain.rl.learners.blackboxoptimizers.evolution.population import Population, SimplePopulation
from pybrain.rl.learners.blackboxoptimizers.evolution.filter     import Randomization
from individual import EvolinoIndividual, EvolinoSubIndividual

from copy   import copy
from random import randrange


class EvolinoPopulation(Population):
    """ Evolino's population class.

        EvolinoIndividuals aren't stored directly, but there is a list of
        subpopulations.
        These subpopulations are used to generate EvolinoIndividuals on demand.

        On initialization, a prototype individual must be supplied. Its genome
        should be a list of chromosomes. A chromosome should be a list of floats.

        A subpopulation of size sub_population_size is created for each of these
        chromosomes.
    """
    def __init__(self, individual, sub_population_size=20, weight_randomizer=Randomization(-0.1,0.1), **kwargs):
        """ @param individual: A prototype individual which is used to determine
                               the structure of the genome.
            @param sub_population_size: integer describing the size of the subpopulations
        """
        Population.__init__(self)

        self._sub_populations = []
        self._sub_population_size = sub_population_size
        self._n_combinations      = 1
        self._verbosity=0
        self.setArgs(**kwargs)
        genome = individual.getGenome()
        for chromosome in genome:
#            print
#            print "=== creating sub population for chrom:", chromosome
            self._sub_populations.append( EvolinoSubPopulation(chromosome, self._sub_population_size, weight_randomizer) )

    def setArgs(self,**kwargs):
        for key, value in kwargs.items():
#            if key in ('nc', 'n_combinations'):
#                self._n_combinations = value
            if key in ("verbose", "verbosity", "ver", "v"):
                self._verbosity = value
            else:
                pass

    def getIndividuals(self):
        """ Returns a set of individuals of type EvolinoIndividual. The individuals
            are generated on the fly. Note that each subpopulation has the same size.
            So the number of resulting EvolinoIndividuals is sub_population_size,
            since each chromosome of each subpopulation will be assembled once.

            The subpopulation container is a sequence with strict order. This
            sequence is iterated sub_population_size times. In each iteration
            one random EvolinoSubIndividual is taken from each sub population.
            After each iteration the resulting sequence of sub individuals
            is supplied to the constructor of a new EvolinoIndividual.
            All EvolinoIndividuals are collected in a set, which is finally returned.
        """
        assert len(self._sub_populations)

        individuals = set()

#        for i in range(self._n_combinations):
        sub_individuals_list = [ list(sp.getIndividuals()) for sp in self._sub_populations ]

        individuals_n = len(sub_individuals_list[0])

        for j in range(individuals_n):
            sub_individual_combination = []
            for sub_individuals in sub_individuals_list:
                sub_individual = sub_individuals.pop( randrange( len( sub_individuals ) ) )
                sub_individual_combination.append( sub_individual )
            individuals.add( EvolinoIndividual(sub_individual_combination) )

        return individuals

    def getSubPopulations(self):
        """ Returns a shallow copy of the list of subpopulation. """
        return copy( self._sub_populations )



    def setIndividualFitness(self, individual, fitness):
        """ The fitness value is not stored directly inside this population,
            but is propagated to the subpopulations of all the subindividuals
            of which the individual consists of.
            The fitness is added to the previous value of the subindividual.
            To reset these values use clearFitness().
        """
        sub_individuals = individual.getSubIndividuals()
        for i,sp in enumerate(self._sub_populations):
            sp.addIndividualFitness( sub_individuals[i], fitness )


    def clearFitness(self):
        """ Clears all fitness values of all subpopulations. """
        for sp in self._sub_populations:
            sp.clearFitness()




class EvolinoSubPopulation(SimplePopulation):
    """ The class for Evolino subpopulations. Mostly the same as SimplePopulation
        but with a few extensions.
        It contains a set of EvolinoSubIndividuals.

        On initialization, a prototype individual is created from the prototype
        chromosome. This individual is then cloned and added so that the
        population exists of max_individuals_n individuals.

        The genomes of these clones are then randomized by the Randomization
        operator.
    """
    def __init__(self, chromosome, max_individuals_n, weight_randomizer=Randomization(-0.1,0.1) ):
        """ @param chromosome: The prototype chromosome
            @param max_individuals_n: The maximum allowed number of individuals
        """
        SimplePopulation.__init__(self)
        self._max_individuals_n = max_individuals_n

        prototype = EvolinoSubIndividual(chromosome)
        for i in range(self._max_individuals_n):
            self.addIndividual(prototype.copy())
        weight_randomizer.apply(self)

    def getMaxIndividualsN(self):
        """ Returns the maximum allowed number of individuals """
        return self._max_individuals_n

    def addIndividualFitness(self, individual, fitness):
        """ Add fitness to the individual's fitness value.
            @param fitness: a float value denoting the fitness
        """
        self._fitness[individual] += fitness




