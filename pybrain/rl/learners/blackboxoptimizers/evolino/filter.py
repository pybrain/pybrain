
__author__ = 'Michael Isik'

from pybrain.rl.learners.blackboxoptimizers.evolution.filter     import Filter, SimpleMutation
from pybrain.rl.learners.blackboxoptimizers.evolution.variate    import CauchyVariate, GaussianVariate
from pybrain.rl.learners.blackboxoptimizers.evolution.population import SimplePopulation
from pybrain.tools.validation import Validator, ModuleValidator

from numpy import zeros, empty, array, dot
from scipy.linalg import pinv2
from copy import deepcopy





class EvolinoEvaluation(Filter):
    """ Evaluate all individuals of the Evolino population, and store their
        fitness value into the population.

        A NetworkWrapper containing the test network must be supplied,
        in order to run tests on the individuals.
    """
    def __init__(self, network_wrapper, dataset, **kwargs):
        """ @param network_wrapper: an instance of NetworkWrapper()
            @param dataset: the evaluation dataset
            @param kwargs: See setArgs() method documentation
        """
        Filter.__init__(self)
        self._verbosity = 0
        self.setArgs( net=network_wrapper, ds=dataset )
        self._evalfunc = None
        self.setArgs( **kwargs )


    def setArgs(self,**kwargs):
        """ @param **kwargs:
                net      : set the network wrapper
                ds       : set the evaluation dataset
                evalfunc : Evaluation function. Will be called with a module
                           and a dataset. Should return the modules fitness value
                           on the dataset.
                v        : set verbosity
        """
        for key, value in kwargs.items():
            if key in ("net"):
                self._network_wrapper = value
            elif key in ("ds", 'data', 'dataset'):
                self._dataset = value
            elif key in ("evalfunc"):
                self._evalfunc = value
            elif key in ("verbose", "verbosity", "ver", "v"):
                self._verbosity = value
            else:
                pass

    def apply(self, population):
        """ Evaluate each individual, and store fitness into EvolinoPopulation.
            Also calculate and set the weight matrix W of the full connection
            between the last hidden layer and the output layer.
        """
        net_wrap = self._network_wrapper
        net      = net_wrap.network
        dataset = self._dataset
        population.clearFitness()
        best_individual = None
        best_W = None
        best_fitness = float('-inf')
        # iterate all individuals. Note, that these individuals are created on the fly
        for individual in population.getIndividuals():
            # calculate phi, which is a list of raw outputs of the net
            # raw output means, that the output wasn't yet forwarded through
            # the last fullconnection between last hidden layer and output layer
            phi = []
            T   = []
            # load the individual's genome into the weights of the net
            net_wrap.setGenome( individual.getGenome() )
            # iterate through all sequences
            for i in range(dataset.getNumSequences()):
                # process next sequence
                seq = dataset.getSequence(i)
                backprojection = zeros(dataset.outdim)
                net.reset()
                for j in range(len(seq[0])):
                    input  = seq[0][j]
                    target = seq[1][j]
                    # backproject the target of the last iteration
                    net_wrap.injectBackproject(backprojection)
                    net.activate(input)
                    raw_out = net_wrap._getRawOutput()

                    phi.append( raw_out )
                    T.append( target)
                    backprojection = target



            # === calculate weight matrix W
            # The output Y of the whole network is calculated by: W*phi
            # Recall, that phi is the matrix containing the outputs of the
            # last hidden layer. Each column will contain the output of one sample.

            # W is the weight matrix of the full connection between
            # the last hidden and the output layer.
            # Each row inside W will contain the input weights of a neuron
            # of the output layer.

            # We want the output of the whole network to be mostly like T,
            # so we want to solve W * phi = T ,
            # which can also be written as W = T * phi^-1 .
            # It's unlikely that an inverse of phi exists, so we use the
            # Moore-Penrose pseudo-inverse which minimizes the summed squared error

            # from now on, each column inside phi is the raw output of a sample
            phi      = array(phi).T
            # calculate the pseudo inverse of phi
            phi_pinv = pinv2(phi)
            # from now on, each column inside T is the target of a sample
            T        = array(T).T
            W        = dot(T, phi_pinv)
# zzzztttt           net_wrap.setOutputWeightMatrix(W)
            net_wrap.setOutputWeightMatrix(W)


            # calculate Y with new weight matrix W
            Y = dot( W, phi ).T


#            Y_dummy = dot(net.getOutputWeightMatrix(),phi).T

            # calculate fitness, which is the negative MSE
            importance = None
            if dataset.data.has_key('importance'):
                importance = dataset.getField('importance')
            if self._evalfunc is not None:
                fitness = self._evalfunc(net, dataset)
            else:
#                fitness = - Validator.MSE(Y,T.T,importance)
                fitness = - ModuleValidator.MSE(net, dataset)

            # set the individual fitness
            if self._verbosity > 1:
                print "Calculated fitness for individual", id(individual), " is ", fitness
            population.setIndividualFitness(individual, fitness)

            if best_fitness < fitness:
                best_fitness = fitness
                best_genome  = deepcopy(individual.getGenome())
                best_W       = deepcopy(W)

        net_wrap.setGenome( best_genome )
        net_wrap.setOutputWeightMatrix(best_W)

#        fitness = - ModuleValidator.MSE(net, dataset)
#        print "AAAAA WINNER FITNESS", fitness


class EvolinoSelection(Filter):
    """ Evolino's selection operator """
    def __init__(self):
        Filter.__init__(self)

    def apply(self, population):
        """ The subpopulations of the EvolinoPopulation are iterated and forwarded
            to the EvolinoSubSelection() operator.
            @param population: object of type EvolinoPopulation
        """
        sps = population.getSubPopulations()
        selection = EvolinoSubSelection()
        for sp in sps:
            selection.apply(sp)




class EvolinoReproduction(Filter):
    """ Evolino's reproduction operator """
    def __init__(self, **kwargs):
        """ @param **kwargs: will be forwarded tho the EvolinoSubReproduction constructor
        """
        Filter.__init__(self)
        self._kwargs = kwargs


    def apply(self, population):
        """ The subpopulations of the EvolinoPopulation are iterated and forwarded
            to the EvolinoSubReproduction() operator.
            @param population: object of type EvolinoPopulation
        """
        sps = population.getSubPopulations()
        reproduction = EvolinoSubReproduction(**self._kwargs)
        for sp in sps:
            reproduction.apply(sp)



# ==================================================== SubPopulation related ===



class EvolinoSubSelection(Filter):
    """ Selection operator for EvolinoSubPopulation objects """
    def __init__(self):
        Filter.__init__(self)

    def apply(self, population):
        """ Simply erases the lowest ranking quarter of individuals from the
            population.
        """
        n = population.getIndividualsN()

        worst = population.getWorstNIndividuals(n/4)
        population.removeIndividuals(worst)





class EvolinoSubReproduction(Filter):
    """ Reproduction operator for EvolinoSubPopulation objects.
    """
    def __init__(self, **kwargs):
        """ @param kwargs: See setArgs() method documentation
        """
        Filter.__init__(self)
        self._mutation_variate = None
        self._sub_population_mutation = EvolinoSubMutation()
        self.mutation = EvolinoSubMutation()
        self._verbosity=0
        self.setArgs(**kwargs)


    def setArgs(self,**kwargs):
        """ @param **kwargs:
                mutation  : set the mutation object
                mv        : set an alternative mutation variate
                verbosity : set verbosity
        """
        for key, value in kwargs.items():
            if key in ('mutation'):
                self._sub_population_mutation = value
            elif key in ('mv', 'mutation-variate'):
                self._sub_population_mutation.setArgs(mv=value)
            elif key in ("verbose", "verbosity", "ver", "v"):
                self._verbosity = value
            else:
                pass

    def apply(self, population):
        """ First determines the number of individuals to be created.
            Then clones the fittest individuals (=parents), mutates these clones
            and adds them to the population.
        """
        max_n     = population.getMaxIndividualsN()
        n         = population.getIndividualsN()
        freespace = max_n - n

        best = population.getBestNIndividuals(freespace)
        children=set()
        for parent in best:
            children.add( parent.copy() )


        dummy_population = SimplePopulation()
        dummy_population.addIndividuals(children)
        self._sub_population_mutation.apply(dummy_population)
        population.addIndividuals(dummy_population.getIndividuals())

        inds = population.getSortedIndividualList()


class EvolinoSubMutation(SimpleMutation):
    """ Mutation operator for EvolinoSubPopulation objects.
        Like SimpleMutation, except, that CauchyVariate is used by default.
    """
    def __init__(self, **kwargs):
        self._mutation_variate = CauchyVariate()
        self._mutation_variate.alpha = 0.1
        SimpleMutation.__init__(self, **kwargs)




