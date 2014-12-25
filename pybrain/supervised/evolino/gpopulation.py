__author__ = 'Michael Isik'

from numpy import Infinity


class Population:
    """ Abstract template for a minimal Population.
        Implement just the methods you need.
    """
    def __init__(self):pass
    def getIndividuals(self):
        """ Should return a shallow copy of the individuals container, so that
            individuals can be manipulated, but not the set of individuals itself.
            For removing or appending individuals to the population, use methods
            like removeIndividual() or addIndividual().
        """
        raise NotImplementedError()

    def addIndividual(self, individual):
        """ Should add an individual to the individual container.
        """
        raise NotImplementedError()

    def addIndividuals(self, individuals):
        """ Should add a set of individuals.
        """
        raise NotImplementedError()

    def removeIndividual(self, individual):
        """ Should remove an individual from the individual container.
        """
        raise NotImplementedError()

    def removeIndividuals(self, individuals):
        """ Should remove a set of individuals.
        """
        raise NotImplementedError()

    def setIndividualFitness(self, individual, fitness):
        """ Should associate the fitness value to the specified individual.
        """
        raise NotImplementedError()

    def getIndividualFitness(self, individual):
        """ Should return the associated fitness value of the specified individual.
        """
        raise NotImplementedError()


class SimplePopulation(Population):
    """ A simple implementation of the abstract Population class.
        Sets are used as individual container. The fitness values are
        stored in a separate dictionary, which maps individuals to fitness values.
        For descriptions of the methods please refer to the Population documentation.
    """
    def __init__(self):
        self._individuals = set()
#        self._fitness = collections.defaultdict( lambda: 0. )
#        self._fitness = collections.defaultdict( lambda: -Infinity )
        self._fitness = {}

    def getIndividuals(self):
        return self._individuals.copy()

    def addIndividual(self, individual):
        self._individuals.add(individual)
        self._fitness[individual] = -Infinity

    def addIndividuals(self, individuals):
        for individual in individuals:
            self.addIndividual(individual)
#        self._individuals = self._individuals.union(set(individuals))

    def removeIndividual(self, individual):
        self._individuals.discard(individual)
        del self._fitness[individual]
#        self._fitness[individual] = -Infinity
#        if self._fitness.has_key(individual):
#            self._fitness[individual] = -Infinity
#            del self._fitness[individual]

    def removeIndividuals(self, individuals):
        for individual in individuals:
            self.removeIndividual(individual)
#        self._individuals.difference_update(set(individuals))


    def setIndividualFitness(self, individual, fitness):
        self._fitness[individual] = fitness

    def getIndividualFitness(self, individual):
#        assert self._fitness.has_key(individual)
        return self._fitness[individual]


    def clearFitness(self):
        """ Clears all stored fitness values """
        for (ind, _) in self._fitness.items():
            self._fitness[ind] = -Infinity
#        self._fitness.clear()

    def getFitnessMap(self):
        """ Returns the fitness dictionary """
        return self._fitness.copy()

    def getMaxFitness(self):
        """ Returns the maximal fitness value """
        return self.getIndividualFitness(self.getBestIndividuals(1))



    def getBestIndividuals(self, n):
        """ Returns n individuals with the highest fitness ranking.
            If n is greater than the number of individuals inside the population
            all individuals are returned.
        """
        return set(self.getBestIndividualsSorted(n))

    def getBestIndividualsSorted(self, n):
        return self.getSortedIndividualList()[:n]


    def getWorstIndividuals(self, n):
        """ Returns the n individuals with the lowest fitness ranking.
            If n is greater than the number of individuals inside the population
            all individuals are returned.
        """
        return set(self.getSortedIndividualList()[-n:])

    def removeWorstIndividuals(self, n):
        """ Removes the n individuals with the lowest fitness ranking.
            If n is greater than the number of individuals inside the population
            all individuals are removed.
        """
        inds = self.getWorstIndividuals(n)
        self.removeIndividuals(inds)


    def getSortedIndividualList(self):
        """ Returns a sorted list of all individuals with descending fitness values. """
        fitness = self._fitness
        return sorted(iter(fitness.keys()), key=lambda k:-fitness[k])


    def getIndividualsN(self):
        """ Returns the number of individuals inside the population """
        return len(self._individuals)

    def getAverageFitness(self):
        return sum(self._fitness.values()) / float(len(self._fitness))




