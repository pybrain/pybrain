
__author__ = 'Michael Isik'

from pybrain.rl.learners.blackboxoptimizers.evolution.filter     import Filter, SimpleMutation
from pybrain.rl.learners.blackboxoptimizers.evolution.variate    import CauchyVariate, GaussianVariate
from pybrain.rl.learners.blackboxoptimizers.evolution.population import SimplePopulation
from pybrain.tools.validation import Validator, ModuleValidator

from numpy import zeros, empty, array, dot, append, concatenate
from scipy.linalg import pinv2
from copy import deepcopy





class EvolinoEvaluation(Filter):
    """ Evaluate all individuals of the Evolino population, and store their
        fitness value into the population.

        A NetworkWrapper containing the test network must be supplied,
        in order to run tests on the individuals.
    """
    def __init__(self, evolino_network, dataset, **kwargs):
        """ @param network_wrapper: an instance of NetworkWrapper()
            @param dataset: the evaluation dataset
            @param kwargs: See setArgs() method documentation
        """
        Filter.__init__(self)
        self.max_fitness = float('-inf')
        self.verbosity = 0
        self.network = evolino_network
        self.dataset = dataset
        self.evalfunc = None
        self.wtvRatio = array([1,1,1], float)
#        self.setArgs( **kwargs )

        for key, val in kwargs.iteritems():
            getattr(self, key)
            setattr(self, key, val)




#    def setArgs(self,**kwargs):
#        """ @param **kwargs:
#                net      : set the network wrapper
#                ds       : set the evaluation dataset
#                evalfunc : Evaluation function. Will be called with a module
#                           and a dataset. Should return the modules fitness value
#                           on the dataset.
#                v        : set verbosity
#        """
#        for key, value in kwargs.items():
#            if key in ("net"):
#                self._network = value
#            elif key in ("ds", 'data', 'dataset'):
#                self._dataset = value
#            elif key in ("evalfunc"):
#                self._evalfunc = value
#            elif key in ("verbose", "verbosity", "ver", "v"):
#                self.verbosity = value
#            else:
#                pass

    def apply(self, population):
        """ Evaluate each individual, and store fitness into EvolinoPopulation.
            Also calculate and set the weight matrix W of the full connection
            between the last hidden layer and the output layer.
        """
        net = self.network
        dataset = self.dataset
        population.clearFitness()
        best_individual = None
        best_W = None
        best_fitness = float('-inf')
        ratio = array(self.wtvRatio,float) / sum(self.wtvRatio)
#        print data_length, training_start, validation_start; exit()


        # iterate all individuals. Note, that these individuals are created on the fly
        for individual in population.getIndividuals():
            # calculate phi, which is a list of raw outputs of the net
            # raw output means, that the output wasn't yet forwarded through
            # the last fullconnection between last hidden layer and output layer
            phi = []
            T   = []
            # load the individual's genome into the weights of the net
            net.setGenome( individual.getGenome() )
#            collected_phi    = array([])
#            collected_target = array([])
            collected_phi    = None
            collected_target = None
            # iterate through all sequences
            for i in range(dataset.getNumSequences()):
                seq = dataset.getSequence(i)
                input  = seq[0]
                target = seq[1]

                training_start   = int(   ratio[0]              * len(input) )
                validation_start = int( ( ratio[0] + ratio[1] ) * len(input) )

                washout_input     = input  [                  : training_start   ]
                washout_target    = target [                  : training_start   ]
                training_input    = input  [ training_start   : validation_start ]
                training_target   = target [ training_start   : validation_start ]
                validation_input  = input  [ validation_start :                  ]
                validation_target = target [ validation_start :                  ]


                # reset
                net.reset()

                # washout
                net._washout(washout_input, washout_target)


                # collect training data
                phi = net._washout(training_input, training_target)
                if collected_phi is not None:
                    collected_phi = append( collected_phi, phi, axis=0 )
                else:
                    collected_phi = phi

                if collected_target is not None:
                    collected_target = append( collected_target, training_target, axis=0 )
                else:
                    collected_target = training_target



            # calculate and set the weight matrix
            collected_phi = collected_phi.T
#            print collected_phi ; exit()
            phi_pinv = pinv2(collected_phi)
            collected_target = collected_target.T
            W                = dot(collected_target, phi_pinv)
            net.setOutputWeightMatrix(W)

            input, output, target = net.calculateOutput( dataset, (ratio[0]+ratio[1], ratio[2]) )


            # calculate fitness, which is the negative MSE
            fitness = - Validator.MSE(output,target)

            # set the individual fitness
            if self.verbosity > 1:
                print "Calculated fitness for individual", individual.getId(), " is ", fitness
            population.setIndividualFitness(individual, fitness)

            if best_fitness < fitness:
                best_fitness = fitness
                best_genome  = deepcopy(individual.getGenome())
                best_W       = deepcopy(W)

        net.reset()
        net.setGenome( best_genome )
        net.setOutputWeightMatrix(best_W)


        # store fitness maximum to use it for triggering burst mutation
#        if self.max_fitness < best_fitness:
        self.max_fitness = best_fitness





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
        """ @param **kwargs: will be forwarded to the EvolinoSubReproduction constructor
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


class EvolinoBurstMutation(Filter):
    """ The burst mutation operator for evolino """
    def __init__(self, **kwargs):
        """ @param **kwargs: will be forwarded to the EvolinoSubReproduction constructor
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
            n_toremove = sp.getIndividualsN()-1
            sp.removeWorstIndividuals(n_toremove)
            reproduction = EvolinoSubReproduction(**self._kwargs)
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

        population.removeWorstIndividuals(n/4)
#        worst = population.getWorstIndividuals(n/4)
#        population.removeIndividuals(worst)





class EvolinoSubReproduction(Filter):
    """ Reproduction operator for EvolinoSubPopulation objects.
    """
    def __init__(self, **kwargs):
        """ @param kwargs: See setArgs() method documentation
        """
        Filter.__init__(self)
        self.mutationVariate = None
        self.mutation = EvolinoSubMutation()
#        self.mutation = EvolinoSubMutation()
        self.verbosity=0
        for key, val in kwargs.iteritems():
            getattr(self, key)
            setattr(self, key, val)

        if self.mutationVariate is not None:
            self.mutation.mutationVariate = self.mutationVariate

#    def setArgs(self,**kwargs):
#        """ @param **kwargs:
#                mutation  : set the mutation object
#                mv        : set an alternative mutation variate
#                verbosity : set verbosity
#        """
#        for key, value in kwargs.items():
#            if key in ('mutation'):
#                self.mutation = value
#            elif key in ('mv', 'mutation-variate'):
#                self.mutation.setArgs(mv=value)
#            elif key in ("verbose", "verbosity", "ver", "v"):
#                self.verbosity = value
#            else:
#                pass

    def apply(self, population):
        """ First determines the number of individuals to be created.
            Then clones the fittest individuals (=parents), mutates these clones
            and adds them to the population.
        """
        max_n     = population.getMaxNIndividuals()
        n         = population.getIndividualsN()
        freespace = max_n - n

        best = population.getBestIndividuals(freespace)
        children=set()
        while True:
            for parent in best:
                children.add( parent.copy() )
                if len(children)>=freespace: break
            if len(children)>=freespace: break

        dummy_population = SimplePopulation()
        dummy_population.addIndividuals(children)
        self.mutation.apply(dummy_population)
        population.addIndividuals(dummy_population.getIndividuals())

        assert population.getMaxNIndividuals() == population.getIndividualsN()

#        inds = population.getSortedIndividualList()







class EvolinoSubMutation(SimpleMutation):
    """ Mutation operator for EvolinoSubPopulation objects.
        Like SimpleMutation, except, that CauchyVariate is used by default.
    """
    def __init__(self, **kwargs):
        self.mutationVariate = CauchyVariate()
        self.mutationVariate.alpha = 0.1
        SimpleMutation.__init__(self, **kwargs)






# ================================================================= playground zzzz



import collections

class EvolinoReproduction2(EvolinoReproduction):
    def __init__(self,**kwargs):
        EvolinoReproduction.__init__(self,**kwargs)
        self._sub_reproduction_map = collections.defaultdict(
            lambda: EvolinoSubReproduction2(**self._kwargs) )



    def apply(self, population):
        sps = population.getSubPopulations()
#        reproduction = EvolinoSubReproduction2(**self._kwargs)
#        for sp in sps:
#            reproduction.apply(sp)

        for sp in sps:
            print "reproduction: ", id(self._sub_reproduction_map[sp])
            self._sub_reproduction_map[sp].apply(sp)
#            reproduction.apply(sp)



class EvolinoSubReproduction2(EvolinoSubReproduction):
    def __init__(self, **kwargs):
        EvolinoSubReproduction.__init__(self)

        self._best_individual = None
        self._best_fitness = float('-inf')




    def apply(self, population):
        """ First determines the number of individuals to be created.
            Then clones the fittest individuals (=parents), mutates these clones
            and adds them to the population.
        """
        best_individual, = population.getBestIndividuals(1)



#        print best_individual in population.getIndividuals();

        best_fitness = population.getIndividualFitness(best_individual)
        if self._best_fitness < best_fitness:
            self._best_individual = best_individual
            self._best_fitness = best_fitness
#            population.addIndividual(self._best_individual.copy())
#        population.addIndividual(self._best_individual.copy())
#        print best
#        exit()
#        print "best individual: ", id(self._best_individual)


#        print best_individual in population.getIndividuals();
#        print self._best_individual is not None and  self._best_individual not in population.getIndividuals()
        doaddbest = self._best_individual is not None and  self._best_individual not in population.getIndividuals()
#        exit()

        reserved_space = 0
        if doaddbest: reserved_space = 1


        max_n     = population.getMaxNIndividuals()
        n         = population.getIndividualsN()
        freespace = max_n - n - reserved_space

        best = population.getBestIndividuals(freespace)
        children=set()
        while True:
            for parent in best:
                children.add( parent.copy() )
                if len(children)>=freespace: break
            if len(children)>=freespace: break

        dummy_population = SimplePopulation()
        dummy_population.addIndividuals(children)
        self.mutation.apply(dummy_population)
        population.addIndividuals(dummy_population.getIndividuals())

#        print "AAAA 1"
        if doaddbest:
#            print "AAAADDDD"
#            print self._best_individual.id
            population.addIndividual(self._best_individual)

        for i in population.getIndividuals():
            print i.id, " ",
        print



        assert population.getMaxNIndividuals() == population.getIndividualsN()

#        inds = population.getSortedIndividualList()




