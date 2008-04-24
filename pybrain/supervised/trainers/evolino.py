
__author__ = 'Michael Isik'



from trainer import Trainer

#from pybrain.rl.learners.blackboxoptimizers.evolino.network    import EvolinoNetwork
from pybrain.rl.learners.blackboxoptimizers.evolino.population     import EvolinoPopulation
from pybrain.rl.learners.blackboxoptimizers.evolino.individual     import EvolinoSubIndividual
from pybrain.rl.learners.blackboxoptimizers.evolino.filter         import EvolinoEvaluation, EvolinoSelection, EvolinoReproduction, EvolinoBurstMutation
from pybrain.rl.learners.blackboxoptimizers.evolution.filter       import Randomization
from pybrain.rl.learners.blackboxoptimizers.evolution.variate      import CauchyVariate, GaussianVariate
from pybrain.rl.learners.blackboxoptimizers.evolino.networkwrapper import NetworkWrapper


from numpy import array

# zzzzzzzzzzttttttttttt
#from pybrain.rl.learners.blackboxoptimizers.evolino.population     import EvolinoPopulation2
#from pybrain.rl.learners.blackboxoptimizers.evolino.filter         import EvolinoReproduction2
#EvolinoPopulation      = EvolinoPopulation2
#EvolinoReproduction    = EvolinoReproduction2


class EvolinoTrainer(Trainer):
    """ The Evolino trainer class.

        Use a network as module that should be trained. There are some restrictions
        the network must follow. Basically, it should be a simple lstm network.
        For more details on these restrictions read NetworkWrapper's documentaion.
    """
    def __init__(self, evolino_network, dataset, **kwargs):
        """ @param network: Network to be evolved
            @param dataset: dataset for evaluating the fitness.
                            Use SequenceDataset or ImportanceDataset
            @param kwargs: See setArgs() method documentation
        """
        Trainer.__init__(self, evolino_network)

        # setting obligatory attributes
        self.network = evolino_network
        self.setData(dataset)

        # === set default arguments ===
        self.subPopulationSize = 8
        self.nCombinations = 4
        self.initialWeightRange = ( -0.1, 0.1 )

        self.mutationAlpha =  0.01
        self.mutationVariate = CauchyVariate(0, self.mutationAlpha)
#        self.mutationVariate  = GaussianVariate(0,0.1)

        self.nBurstMutationEpochs = 20
        self.verbosity = 0

        self.evalfunc  = None
        self.wtvRatio = (1,1,1)

        self.burstMutation = EvolinoBurstMutation()
        self.backprojectionFactor = float(evolino_network.backprojectionFactor)

        # === overwrite default arguments with kwargs ===
        for key, val in kwargs.iteritems():
            getattr(self, key)
            setattr(self, key, val)


        # === create and modify objects ===
        self.mutationVariate.alpha = self.mutationAlpha

        evolino_network.backprojectionFactor = float(self.backprojectionFactor)

        self._population = EvolinoPopulation(
            EvolinoSubIndividual( evolino_network.getGenome() ),
            self.subPopulationSize,
            self.nCombinations,
            Randomization(
                self.initialWeightRange[0],
                self.initialWeightRange[1])
            )
        self._population.nCombinations = self.nCombinations



        filters = []
        self._evaluation   = EvolinoEvaluation(evolino_network, self.ds, evalfunc=self.evalfunc, wtvRatio=self.wtvRatio, verbosity=self.verbosity)
        self._selection    = EvolinoSelection()
        self._reproduction = EvolinoReproduction( mutationVariate=self.mutationVariate)

        filters.append( self._evaluation   )
        filters.append( self._selection    )
        filters.append( self._reproduction )
#        filters.append( EvolinoReproduction2( mv=self.mutationVariate) )

        self._filters = filters

        self.totalepochs = 0
        self._max_fitness = self._evaluation.max_fitness
        self._max_fitness_epoch = self.totalepochs







#    def setArgs(self, **kwargs):
#        for key, val in kwargs.iteritems():
#            setattr(self, key, val)




#    def setArgs(self,**kwargs):
#        """ @param **kwargs:
#                sps      : size of subpopulations
#                mv       : the variate used for mutations needed for replication
#                evalfunc : Evaluation function. Will be called with a module
#                           and a dataset. Should return the modules fitness value
#                           on the dataset.
#                v        : set verbosity
#        """
#        for key, value in kwargs.items():
#            if   key in ('sps','subPopulationSize'):
#                self.subPopulationSize = value
#            elif key in ('mv','mutation_variate'):
#                self.mutationVariate = value
#            elif key in ('ma','mutation_alpha'):
#                self.mutationAlpha = value
#            elif key in ('iwr','initial_weight_range'):
#                self.initialWeightRange = value
#            elif key in ('bme','burstMutation_epochs'):
#                self.nBurstMutationEpochs = value
#            elif key in ('evalfunc'):
#                self.evalfunc = value
#            elif key in ('wtvRatio', 'washout_training_validation_ratio'):
#                self.wtvRatio = value
#            elif key in ('n_combinations'):
#                self.nCombinations = value
#            elif key in ('verbose', 'verbosity', 'ver', 'v'):
#                self.verbosity = value
#            else: pass


    def trainOnDataset(self,*args,**kwargs):
        """ Not implemented """
        raise NotImplementedError()


    def train(self):
        """ Evolve for one epoch. """
        self.totalepochs += 1

        if self.totalepochs - self._max_fitness_epoch >= self.nBurstMutationEpochs:
            if self.verbosity: print "RUNNING BURST MUTATION"
            self.burstMutate()
            self._max_fitness_epoch = self.totalepochs


        for filter in self._filters:
            filter.apply( self._population )

        if self._max_fitness < self._evaluation.max_fitness:
            if self.verbosity: print "GAINED FITNESS: ", self._max_fitness, " -->" ,self._evaluation.max_fitness, "\n"
            self._max_fitness = self._evaluation.max_fitness
            self._max_fitness_epoch = self.totalepochs
        else:
            if self.verbosity: print "DIDN'T GAIN FITNESS:", "best =", self._max_fitness, "    current-best = ", self._evaluation.max_fitness, "\n"



#        print self._network_wrapper.getGenome()

    def burstMutate(self):
        self.burstMutation.apply(self._population)
#        EvolinoBurstMutation().apply(self._population)







