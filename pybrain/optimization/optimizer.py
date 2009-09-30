__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import array, randn, ndarray, isinf, isnan, isscalar
import logging

from pybrain.utilities import setAllArgs, abstractMethod, DivergenceError
from pybrain.rl.learners.learner import PhylogeneticLearner
from pybrain.rl.learners.directsearch.directsearch import DirectSearch
from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.rl.environments.functions.function import FunctionEnvironment
from pybrain.rl.environments.fitnessevaluator import FitnessEvaluator
from pybrain.rl.environments.functions.transformations import oppositeFunction


class BlackBoxOptimizer(DirectSearch, PhylogeneticLearner):
    """ The super-class for learning algorithms that treat the problem as a black box. 
    At each step they change the policy, and get a fitness value by invoking 
    the FitnessEvaluator (provided as first argument upon initialization).
    
    Evaluable objects can be lists or arrays of continuous values (also wrapped in ParameterContainer) 
    or subclasses of Evolvable (that define its methods).
    """    
    
    # Minimize or maximize fitness? By default, all algorithms maximize it.
    minimize = False

    # Is there a known value of sufficient fitness?
    desiredEvaluation = None    

    # dimension of the search space, if applicable
    numParameters = None
    
    # Bookkeeping settings
    storeAllEvaluations = False
    storeAllEvaluated = False
    
    # an optimizer can take different forms of evaluables, and depending on its
    # needs, wrap them into a ParameterContainer (which is also an Evolvable)
    # or unwrap them to act directly on the array of parameters (all ContinuousOptimizers)
    wasWrapped = False
    wasUnwrapped = False
    wasOpposed = False
    
    # stopping criteria
    maxEvaluations = 1e6
    maxLearningSteps = None    
    
    # providing information during the learning
    listener = None
    verbose = False
    
    # some algorithms have a predetermined (minimal) number of 
    # evaluations they will perform during each learningStep:
    batchSize = 1
    
    def __init__(self, evaluator = None, initEvaluable = None, **kwargs):
        """ The evaluator is any callable object (e.g. a lambda function). """
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
            
        if evaluator is not None:
            self.setEvaluator(evaluator, initEvaluable)
        else:
            self.__evaluator = None
        
    def setEvaluator(self, evaluator, initEvaluable = None):
        self.__evaluator = evaluator        

        # default settings, if provided by the evaluator:
        if isinstance(evaluator, FitnessEvaluator):
            if self.desiredEvaluation is None:
                self.desiredEvaluation = evaluator.desiredValue               
            if self.minimize is not evaluator.toBeMinimized:
                logging.info('Algorithm is set to minimize='+str(self.minimize)+\
                            ' but evaluator is set to minimize='+str(evaluator.toBeMinimized)+\
                            '.\n The opposite function will be optimized.')
                self.__evaluator = oppositeFunction(evaluator)
                self.wasOpposed = True
                if self.desiredEvaluation is not None:
                    self.desiredEvaluation *= -1
            if self.numParameters is None:
                # in some cases, we can deduce the dimension from the provided evaluator:
                if isinstance(evaluator, FunctionEnvironment):
                    self.numParameters = evaluator.xdim          
        #set the starting point for optimization (as provided, or randomly)
        self._setInitEvaluable(initEvaluable)        
        self._additionalInit()
        self.bestEvaluable = self._initEvaluable
        self.bestEvaluation = None
        
    def _additionalInit(self):
        """ a method for subclasses that need additional initialization code but don't want to redefine __init__ """

    def _setInitEvaluable(self, evaluable):
        if evaluable is None:
            # if there is no initial point specified, we start at one that's sampled 
            # normally around the origin.
            if self.numParameters is not None:
                evaluable = randn(self.numParameters)
            else:
                raise ValueError('Could not determine the dimensionality of the evaluator. '+\
                                 'Please provide an initial search point.')   
        if isinstance(evaluable, list):
            evaluable = array(evaluable)
        
        # If the evaluable is provided as a list of numbers or as an array,
        # we wrap it into a ParameterContainer.
        if isinstance(evaluable, ndarray):            
            pc = ParameterContainer(len(evaluable))
            pc._setParameters(evaluable)
            self.wasWrapped = True
            evaluable = pc
        self._initEvaluable = evaluable
        if isinstance(self._initEvaluable, ParameterContainer):
            self.numParameters = len(self._initEvaluable)      
    
    def learn(self, additionalLearningSteps = None):
        """ The main loop that does the learning. """
        assert self.__evaluator is not None, "No evaluator has been set. Learning cannot start."
        if additionalLearningSteps is not None:
            self.maxLearningSteps = self.numLearningSteps + additionalLearningSteps
        while not self._stoppingCriterion():
            try:
                self._learnStep()
                self._notify()
                self.numLearningSteps += 1
            except DivergenceError:
                logging.warning("Algorithm diverged. Stopped after "+str(self.numLearningSteps)+" learning steps.")
                break
        return self._bestFound()
        
    def _learnStep(self):
        """ The core method to be implemented by all subclasses. """
        abstractMethod()        
        
    def _bestFound(self):
        """ return the best found evaluable and its associated fitness. """
        bestE = self.bestEvaluable.params.copy() if self.wasWrapped else self.bestEvaluable
        if self.bestEvaluation is None:
            bestF = None
        else:
            bestF = -self.bestEvaluation if (self.wasOpposed and isscalar(self.bestEvaluation)) else self.bestEvaluation
        return bestE, bestF
        
    def _oneEvaluation(self, evaluable):
        """ This method should be called by all optimizers for producing an evaluation. """
        if self.wasUnwrapped:
            self.wrappingEvaluable._setParameters(evaluable)
            res = self.__evaluator(self.wrappingEvaluable)
        elif self.wasWrapped:            
            res = self.__evaluator(evaluable.params)
        else:            
            res = self.__evaluator(evaluable)
        
        self.numEvaluations += 1
        
        if isscalar(res):
            # detect numerical instability
            if isnan(res) or isinf(res):
                raise DivergenceError
                
            # always keep track of the best
            if (self.numEvaluations == 0
                or (self.minimize and res <= self.bestEvaluation)
                or (not self.minimize and res >= self.bestEvaluation)):
                self.bestEvaluation = res
                self.bestEvaluable = evaluable.copy()
        
        # if desired, also keep track of all evaluables and/or their fitness.                        
        if self.storeAllEvaluated:
            if self.wasUnwrapped:            
                self._allEvaluated.append(self.wrappingEvaluable.copy())
            elif self.wasWrapped:            
                self._allEvaluated.append(evaluable.params.copy())
            else:            
                self._allEvaluated.append(evaluable.copy())        
        if self.storeAllEvaluations:
            if self.wasOpposed or not isscalar(res):
                self._allEvaluations.append(res)
            else:
                self._allEvaluations.append(-res)
        return res
    
    def _stoppingCriterion(self):
        if self.maxEvaluations is not None and self.numEvaluations+self.batchSize > self.maxEvaluations:
            return True
        if self.desiredEvaluation is not None and self.bestEvaluation is not None and isscalar(self.bestEvaluation):
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
        if isinstance(evaluable, ParameterContainer):
            self.wrappingEvaluable = evaluable.copy()
            self.wasUnwrapped = True
        elif not (evaluable is None 
                  or isinstance(evaluable, list) 
                  or isinstance(evaluable, ndarray)):
            raise ValueError('Continuous optimization algorithms require a list, array or'+\
                             ' ParameterContainer as evaluable.')
        BlackBoxOptimizer._setInitEvaluable(self, evaluable)
        self.wasWrapped = False
        self._initEvaluable = self._initEvaluable.params.copy()     
        
    def _bestFound(self):
        """ return the best found evaluable and its associated fitness. """
        bestE, bestF = BlackBoxOptimizer._bestFound(self)
        if self.wasUnwrapped:
            self.wrappingEvaluable._setParameters(bestE)
            bestE = self.wrappingEvaluable.copy()
        return bestE, bestF
   
    