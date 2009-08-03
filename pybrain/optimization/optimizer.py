__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import array, randn, ndarray

from pybrain.utilities import setAllArgs, abstractMethod
from pybrain.rl.learners.learner import PhylogeneticLearner
from pybrain.rl.learners.directsearch.directsearch import DirectSearch
from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.rl.environments.functions.function import FunctionEnvironment


class BlackBoxOptimizer(DirectSearch, PhylogeneticLearner):
    """ The super-class for learning algorithms that treat the problem as a black box. 
    At each step they change the policy, and get a fitness value by invoking 
    the fitness-evaluator (provided as first argument upon initialization).
    
    Evaluable objects can be either arrays of continuous values (also wrapped in ParameterContainer) 
    or subclasses of Evolvable (that define its methods).
    """    
    
    # Minimize or maximize fitness?
    minimize = False

    # Is there a known value of sufficient fitness?
    desiredEvaluation = None    

    # dimension of the search space, if applicable
    numParameters = None
    
    # Bookkeeping settings
    storeAllEvaluations = False
    storeAllEvaluated = False
    wasWrapped = False
    wasUnwrapped = False
    
    # stopping criteria
    maxEvaluations = 1e6
    maxLearningSteps = None    
    
    # providing information during the learning
    listener = None
    verbose = False

    def __init__(self, evaluator, initEvaluable = None, **kwargs):
        """ The evaluator is any callable object (e.g. a lambda function). """
        self.__evaluator = evaluator
        # set all algorithm-specific parameters in one go:
        setAllArgs(self, kwargs)
        # bookkeeping
        self.numEvaluations = 0      
        self.numLearningSteps = 0
        if self.storeAllEvaluated:
            self._allEvaluated = []
            self._allEvaluations = []
        elif self.storeAllEvaluations:
            self._allEvaluations = []
        # default settings, if provided by the evaluator:
        if isinstance(evaluator, FunctionEnvironment):
            if self.minimize is not evaluator.toBeMinimized:
                raise TypeError('Algorithm is set to minimize='+str(self.minimize)+\
                                ' but evaluator is set to minimize='+str(evaluator.toBeMinimized))
            if self.numParameters is None:
                self.numParameters = evaluator.xdim
            if self.desiredEvaluation is None:
                self.desiredEvaluation = evaluator.desiredValue             
        #set the starting point for optimization (as provided, or randomly)
        self._setInitEvaluable(initEvaluable)

    def learn(self, additionalLearningSteps = None):
        """ The main loop that does the learning. """
        if additionalLearningSteps is not None:
            self.maxLearningSteps = self.numLearningSteps + additionalLearningSteps
        while not self._stoppingCriterion():
            self._learnStep()
            self._notify()
            self.numLearningSteps += 1
        return self._bestFound()
        
    def _learnStep(self):
        """ The core method to be implemented by all subclasses. """
        abstractMethod()
        
    def _setInitEvaluable(self, evaluable):
        # If the evaluable is provided as a list of numbers or as an array,
        # we wrap it into a ParameterContainer.
        if isinstance(evaluable, list):
            evaluable = array(evaluable)
        if isinstance(evaluable, ndarray):            
            pc = ParameterContainer(len(evaluable))
            pc._setParameters(evaluable)
            self.wasWrapped = True
            self._initEvaluable = pc
        elif evaluable is None:
            raise ValueError('An initial evaluable must be specified to start optimization.')
        else:
            self._initEvaluable = evaluable
        self._oneEvaluation(self._initEvaluable)
        
    def _bestFound(self):
        """ return the best found evaluable and its associated fitness. """
        if self.wasWrapped:
            return self.bestEvaluable.params.copy(), self.bestEvaluation
        else:
            return self.bestEvaluable, self.bestEvaluation
        
    def _oneEvaluation(self, evaluable):
        """ This method should be called by all optimizers for producing an evaluation. """
        if self.wasWrapped:            
            res = self.__evaluator(evaluable.params)
        else:            
            res = self.__evaluator(evaluable)
        # always keep track of the best
        if (self.numEvaluations == 0
            or (self.minimize and res <= self.bestEvaluation)
            or (not self.minimize and res >= self.bestEvaluation)):
            self.bestEvaluation = res
            self.bestEvaluable = evaluable
        self.numEvaluations += 1
        # if desired, also keep track of all others                        
        if self.storeAllEvaluated:
            self._allEvaluated.append(evaluable.copy())
            self._allEvaluations.append(res)
        elif self.storeAllEvaluations:
            self._allEvaluations.append(res)
        return res
    
    def _stoppingCriterion(self):
        if self.maxEvaluations is not None and self.numEvaluations >= self.maxEvaluations:
            return True
        if self.desiredEvaluation is not None:
            if ((self.minimize and self.bestEvaluation <= self.desiredEvaluation)
                or (not self.minimize and self.bestEvaluation >= self.desiredEvaluation)):
                return True
        if self.maxLearningSteps is not None and self.numLearningSteps >= self.maxLearningSteps:
            return True
        return False
    
    def _notify(self):
        """ Provide some feedback during the run. """
        if self.verbose:
            print 'Step:', self.numLearningSteps, 'best:', self.bestEvaluation
        if self.listener is not None:
            self.listener(self.bestEvaluable, self.bestEvaluation)
            
        
class ContinuousOptimizer(BlackBoxOptimizer):
    """ A more restricted class of black-box optimization algorithms
    that assume the parameters to be necessarily an array of continuous values 
    (which can be wrapped in a ParameterContainer). """    
            
    def _setInitEvaluable(self, evaluable):
        """ If the parameters are wrapped, we keep track of the wrapper explicitly. """
        if evaluable is None:
            # if there is no initial point specified, we start at one that's sampled 
            # normally around the origin.
            if self.numParameters is not None:
                evaluable = randn(self.numParameters)
            else:
                raise ValueError('Could not determine the dimensionality of the evaluator,'+\
                                 ' and thus not start the optimization. Please provide an initial search point.')            
        if isinstance(evaluable, ParameterContainer):
            self.wrappingEvaluable = evaluable.copy()
            self.wasUnwrapped = True
            evaluable = evaluable.params.copy()     
        elif isinstance(evaluable, list):
            evaluable = array(evaluable)
        self.numParameters = len(evaluable)
        self._initEvaluable = evaluable
        self._oneEvaluation(self._initEvaluable)
        
    def _oneEvaluation(self, evaluable):        
        if self.wasUnwrapped:
            self.wrappingEvaluable._setParameters(evaluable)
            evaluable = self.wrappingEvaluable.copy()
        return BlackBoxOptimizer._oneEvaluation(self, evaluable)
        
