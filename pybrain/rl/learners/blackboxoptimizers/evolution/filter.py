
__author__ = 'Michael Isik'


from pybrain.rl.learners.blackboxoptimizers.evolution.variate import UniformVariate, GaussianVariate

class Filter(object):
    """ Base class for all kinds of operators on the population during the
        evolutionary process like mutation, selection or evaluation.
    """
    def __init__(self):
        pass
    def apply(self,population):
        """ Applies an operation on a population. """
        raise NotImplementedError()

def isiter(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False



class SimpleGenomeManipulation(Filter):
    """ Abstract filter class for simple genome manipulation. """
    def __init__(self):
        Filter.__init__(self)

    def _manipulateGenome(self, genome, manfunc=None):
        """ Manipulates the genome inplace by calling the abstract _manipulateValue()
            method on each float found.
            @param genome: Arbitrary netsted iterateable container whose leaf
                           elements may be floats or empty containers.
                           E.g. [ [1.] , [1. , 2. , 2 , [3. , 4.] ] , [] ]
            @param manfunc: function that manipulates the found floats.
                            If omitted, self._manipulateValue() is used.
                            See its documentation for the signature description.
        """
        assert isiter(genome)
        if manfunc is None:  manfunc = self._manipulateValue

        for i,v in enumerate(genome):
            if isiter(v):
                self._manipulateGenome(v,manfunc)
            else:
                genome[i] = manfunc(v)

    def _manipulateValue(self, value):
        """ Abstract Method, which should manipulate a value.
            Should return the manipulated value
        """
        raise NotImplementedError()



class SimpleMutation(SimpleGenomeManipulation):
    _mutation_variate = None
    """ A simple mutation filter, which uses a gaussian variate per default
        for mutation.
    """
    def __init__(self, **kwargs):
        """ @param kwargs: See setArgs() method documentation
        """
        SimpleGenomeManipulation.__init__(self)
        if self._mutation_variate is None:
            self._mutation_variate = GaussianVariate()
            self._mutation_variate.alpha = 0.1
        self._verbosity=0
        self.setArgs(**kwargs)


    def setArgs(self,**kwargs):
        """ @param **kwargs:
                mv : set an alternative mutation variate
                v  : set verbosity
        """
        for key, value in kwargs.items():
            if key in ('mv', 'mutation_variate'):
                self._mutation_variate = value
            elif key in ("verbose", "verbosity", "ver", "v"):
                self._verbosity = value
            else: pass

    def apply(self, population):
        """ Apply the mutation to the population
            @param population: must implement the getIndividuals() method
        """
        for individual in population.getIndividuals():
            self._mutateIndividual(individual)

    def _mutateIndividual(self, individual):
        """ Mutate a single individual
            @param individual: must implement the getGenome() method
        """
        genome = individual.getGenome()
        self._manipulateGenome(genome)

    def _manipulateValue(self, value):
        """ Implementation of the abstract method of class SimpleGenomeManipulation
            Set's the x0 value of the variate to value and takes a new sample
            value and returns it.
        """
        self._mutation_variate.x0 = value
        newval = self._mutation_variate.getSample()
#        print "MUTATED: ", value, "--->", newval
        return newval





class Randomization(SimpleGenomeManipulation):
    """ Randomizes the genome of all individuals of a population
        Uses UniformVariate to do so.
    """
    def __init__(self, minval=0., maxval=1.):
        SimpleGenomeManipulation.__init__(self)
        self._minval = minval
        self._maxval = maxval

    def apply(self, population):
        self._uniform_variate = UniformVariate(self._minval,self._maxval)
        for individual in population.getIndividuals():
            self._manipulateGenome(individual.getGenome())

    def _manipulateValue(self, value):
        """ See SimpleGenomeManipulation._manipulateValue() for more information """
        return self._uniform_variate.getSample()





