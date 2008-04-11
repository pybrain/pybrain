
__author__ = 'Michael Isik'



from trainer import Trainer

#from pybrain.rl.learners.blackboxoptimizers.evolino.network    import EvolinoNetwork
from pybrain.rl.learners.blackboxoptimizers.evolino.population     import EvolinoPopulation
from pybrain.rl.learners.blackboxoptimizers.evolino.individual     import EvolinoSubIndividual
from pybrain.rl.learners.blackboxoptimizers.evolino.filter         import EvolinoEvaluation, EvolinoSelection, EvolinoReproduction
from pybrain.rl.learners.blackboxoptimizers.evolution.filter       import Randomization
from pybrain.rl.learners.blackboxoptimizers.evolution.variate      import CauchyVariate, GaussianVariate
from pybrain.rl.learners.blackboxoptimizers.evolino.networkwrapper import NetworkWrapper



class EvolinoTrainer(Trainer):
    """ The Evolino trainer class.

        Use a network as module that should be trained. There are some restrictions
        the network must follow. Basically, it should be a simple lstm network.
        For more details on these restrictions read NetworkWrapper's documentaion.
    """
    def __init__(self, network, dataset, **kwargs):
        """ @param network: Network to be evolved
            @param dataset: dataset for evaluating the fitness.
                            Use SequenceDataset or ImportanceDataset
            @param kwargs: See setArgs() method documentation
        """
        network.sortModules()
        Trainer.__init__(self, network)
        self.setData(dataset)

        # set arguments
        self._sub_population_size = 20
        self._initial_weight_range = ( -0.1, 0.1 )
        self._mutation_alpha =  0.01
        self._mutation_variate = CauchyVariate()
#        self._mutation_variate  = GaussianVariate(0,0.1)
        self._verbosity = 0
        self._evalfunc  = None
        self.setArgs(**kwargs)
        self._mutation_variate.alpha = self._mutation_alpha

        net_wrap = NetworkWrapper(network)
        self._network_wrapper = net_wrap
        self._population = EvolinoPopulation(
            EvolinoSubIndividual( net_wrap.getGenome() ),
            self._sub_population_size,
            Randomization(
                self._initial_weight_range[0],
                self._initial_weight_range[1])
            )

        filters = []
        filters.append( EvolinoEvaluation(net_wrap, self.ds, evalfunc=self._evalfunc, verbosity=self._verbosity) )
        filters.append( EvolinoSelection() )
        filters.append( EvolinoReproduction( mv=self._mutation_variate) )

        self._filters = filters

        self.totalepochs = 0


    def setArgs(self,**kwargs):
        """ @param **kwargs:
                sps      : size of subpopulations
                mv       : the variate used for mutations needed for replication
                evalfunc : Evaluation function. Will be called with a module
                           and a dataset. Should return the modules fitness value
                           on the dataset.
                v        : set verbosity
        """
        for key, value in kwargs.items():
            if   key in ('sps','sub_population_size'):
                self._sub_population_size = value
            elif key in ('mv','mutation_variate'):
                self._mutation_variate = value
            elif key in ('ma','mutation_alpha'):
                self._mutation_alpha = value
            elif key in ('iwr','initial_weight_range'):
                self._initial_weight_range = value
            elif key in ('evalfunc'):
                self._evalfunc = value
            elif key in ('verbose', 'verbosity', 'ver', 'v'):
                self._verbosity = value
            else: pass


    def trainOnDataset(self,*args,**kwargs):
        """ Not implemented """
        raise NotImplementedError()


    def train(self):
        """ Evolve for one epoch. """
        self.totalepochs += 1
        for filter in self._filters:
            filter.apply( self._population )
        print self._network_wrapper.getGenome()









