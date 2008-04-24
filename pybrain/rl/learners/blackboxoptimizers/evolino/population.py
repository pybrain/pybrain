
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

        A subpopulation of size subPopulationSize is created for each of these
        chromosomes.
    """
    def __init__(self, individual, subPopulationSize, nCombinations, valueInitializer=Randomization(-0.1,0.1), **kwargs):
        """ @param individual: A prototype individual which is used to determine
                               the structure of the genome.
            @param subPopulationSize: integer describing the size of the subpopulations
        """
        Population.__init__(self)

        self._subPopulations = []

        # set default values for arguments
#        self.subPopulationSize = subPopulationSize
#        self.valueInitializer = Randomization(-0.1,0.1),
        self.nCombinations = nCombinations
        self.verbosity=0

        # set the passed arguments
        for key, val in kwargs.iteritems():
            getattr(self, key)
            setattr(self, key, val)
#        self.setArgs(**kwargs)



        genome = individual.getGenome()
        for chromosome in genome:
#            valueInitializer._minval-=0.07 # zzzzzttttt
#            valueInitializer._maxval-=0.07 # zzzzzttttt
            self._subPopulations.append(
                EvolinoSubPopulation(chromosome, subPopulationSize, valueInitializer) )

#    def _initSubPopulations(self):
#        for sp in self._subPopulations:
#            sp.setArgs(subPopulationSize=self.subPopulationSize, valueInitializer=self.valueInitializer)




#    def setArgs(self, **kwargs):
#        for key, val in kwargs.iteritems():
#            getattr(self, key)
#            setattr(self, key, val)

#    def setArgs(self,**kwargs):
#        for key, value in kwargs.items():
##            if key in ('nc', 'nCombinations'):
##                self._nCombinations = value
#            if key in ("verbose", "verbosity", "ver", "v"):
#                self.verbosity = value
#            else:
#                pass

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

        for i in range(self.nCombinations):
            subIndividualsList = [ list(sp.getIndividuals()) for sp in self._subPopulations ]

            nIndividuals = len(subIndividualsList[0])

            for j in range(nIndividuals):
                subIndividualCombination = []
                for subIndividuals in subIndividualsList:
                    sub_individual = subIndividuals.pop( randrange( len( subIndividuals ) ) )
                    subIndividualCombination.append( sub_individual )
                individuals.add( EvolinoIndividual(subIndividualCombination) )

        return individuals

    def getSubPopulations(self):
        """ Returns a shallow copy of the list of subpopulation. """
        return copy( self._subPopulations )



    def setIndividualFitness(self, individual, fitness):
        """ The fitness value is not stored directly inside this population,
            but is propagated to the subpopulations of all the subindividuals
            of which the individual consists of.
            The fitness is added to the previous value of the subindividual.
            To reset these values use clearFitness().
        """
        subIndividuals = individual.getSubIndividuals()
        for i,sp in enumerate(self._subPopulations):
            sp.addIndividualFitness( subIndividuals[i], fitness ) # zzzzztttttttttt das original



#        for i,sp in enumerate(self._subPopulations):   # zzzzzzzzzzzztttttttt versuch
#            sub_individual = subIndividuals[i]
##            sp.setIndividualFitness( sub_individual, fitness )
#            old_fitness = sp.getIndividualFitness( sub_individual )
#            if old_fitness < fitness:
#                sp.setIndividualFitness( sub_individual, fitness )


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
    def __init__(self, chromosome, maxNIndividuals, valueInitializer=Randomization(-0.1,0.1), **kwargs):
        """ @param chromosome: The prototype chromosome
            @param maxNIndividuals: The maximum allowed number of individuals
        """
        SimplePopulation.__init__(self)

        self._prototype = EvolinoSubIndividual(chromosome)

        self._maxNIndividuals  = maxNIndividuals
        self._valueInitializer = valueInitializer


#        self.maxNIndividuals = maxNIndividuals
#        self.maxNIndividuals  = property(self._getMaxNIndividuals, self._setMaxNIndividuals)
#        self.valueInitializer = property(self._getValueInitializer, self._setValueInitializer)

        self.setArgs(**kwargs)


#        self._initPopulation()

        for i in range(maxNIndividuals):
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
            @param fitness: a float value denoting the fitness
        """
        self._fitness[individual] += fitness

#    def _setMaxNIndividuals(self, value):
#        self._maxNIndividuals = value
#        self._initPopulation()

#    def _getMaxNIndividuals(self):
#        return self._maxNIndividuals


#    def _setValueInitializer(self, value):
#        self._valueInitializer = value
#        self._initPopulation()

#    def _getValueInitializer(self):
#        return self._valueInitializer



# ============================================== playground


class EvolinoPopulation2(EvolinoPopulation):
    def setIndividualFitness(self, individual, fitness):
        subIndividuals = individual.getSubIndividuals()
        for i,sp in enumerate(self._subPopulations):   # zzzzzzzzzzzztttttttt versuch
            sub_individual = subIndividuals[i]
#            sp.setIndividualFitness( sub_individual, fitness )
            old_fitness = sp.getIndividualFitness( sub_individual )
            if old_fitness < fitness:
                sp.setIndividualFitness( sub_individual, fitness )




