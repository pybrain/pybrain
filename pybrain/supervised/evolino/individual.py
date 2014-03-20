__author__ = 'Michael Isik'


from pybrain.supervised.evolino.gindividual import Individual
from copy import copy, deepcopy


class EvolinoIndividual(Individual):
    """ Individual of the Evolino framework, that consists of a list of
        sub-individuals. The genomes of the sub-individuals are used as
        the cromosomes for the main individual's genome.
        The genome of an individual encodes the RNN's connection weights.
    """
    def __init__(self, sub_individuals):
        """ :key sub_individuals: sequence (e.g. list) of sub-individuals
        """
        self._sub_individuals = list(sub_individuals)

    def getGenome(self):
        """ Returns the genome created by concatenating the chromosomes supplied
            by the sub-individuals.
        """
        genome = []
        for sub_individual in self._sub_individuals:
            genome.append(deepcopy(sub_individual.getGenome()))
        return genome

    def getSubIndividuals(self):
        """ Returns a shallow copy of the list of sub-individuals """
        return copy(self._sub_individuals)



class EvolinoSubIndividual(Individual):
    """ The sub-individual class of evolino
    """
    _next_id = 0
    def __init__(self, genome):
        """ :key genome: Any kind of nested iteratable container containing
                           floats as leafs
        """
        self.setGenome(genome)
        self.id = EvolinoSubIndividual._next_id
        EvolinoSubIndividual._next_id += 1

    def getGenome(self):
        """ Returns the genome. """
        return self._genome

    def setGenome(self, genome):
        """ Sets the genome. """
        self._genome = genome

    def copy(self):
        """ Returns a complete copy of the individual. """
        return copy(self)

    def __copy__(self):
        """ Returns a complete copy of the individual. """
        return EvolinoSubIndividual(deepcopy(self._genome))




