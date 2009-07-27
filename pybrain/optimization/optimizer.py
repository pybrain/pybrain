__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import array, randn

from pybrain.utilities import setAllArgs, abstractMethod
from pybrain.rl.learners.learner import PhylogeneticLearner
from pybrain.rl.learners.directsearch.directsearch import DirectSearch
from pybrain.structure.parametercontainer import ParameterContainer



class BlackBoxOptimizer(DirectSearch, PhylogeneticLearner):
    """ The super-class for learning algorithms that treat the problem as a black box. 
    At each step they change the policy, and get a fitness value by invoking 
    the fitness-evaluator (provided as first argument upon initialization).
    
    Evaluable objects can be either arrays of continuous values (also wrapped in ParameterContainer) 
    or subclasses of Evolvable (that define those methods).
    """    
    
    desiredEvaluation = None
    maxEvaluations = 1e6
    maxLearningSteps = None
    
    # Minimize or maximize fitness?
    minimize = False

    # Bookkeeping settings
    storeAllEvaluations = False
    storeAllEvaluated = False

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
        #set the starting point for optimization (as provided, or randomly)
        self._setInitEvalauable(initEvaluable)

    def learn(self, additionalLearningSteps = None):
        """ The main loop that does the learning. """
        if additionalLearningSteps is not None:
            self.maxLearningSteps = self.numLearningSteps + additionalLearningSteps
        while not self._stoppingCriterion():
            self._learnStep()
        return self._bestFound()
        
    def _learnStep(self):
        """ The core method to be implemented by all subclasses. """
        abstractMethod()
        
    def _setInitEvaluable(self, evaluable):
        # If the evaluable is provided as a list of numbers or as an array,
        # we wrap it into a ParameterContainer.
        if isinstance(evaluable, list) or isinstance(evaluable, array):            
            pc = ParameterContainer(len(evaluable))
            pc._setParameters(evaluable)
            self.initEvaluable = pc
        else:
            self.initEvaluable = evaluable
        self._oneEvaluation(self.initEvaluable)
        
    def _bestFound(self):
        """ return the best found evaluable and its associated fitness. """
        return self.bestEvaluable, self.bestEvaluation
        
    def _oneEvaluation(self, evaluable):
        """ This method should be called by all optimizers for producing an evaluation. """
        res = self.__evaluator(evaluable)
        # always keep track of the best
        if (self.numEvaluations == 0
            or (self.minimize and res < self.bestEvaluation)
            or (not self.minimize and res > self.bestEvaluation)):
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
        if self.desiredEvaluation is not None and self.bestEvaluation >= self.desiredEvaluation:
            return True
        if self.maxLearningSteps is not None and self.numLearningSteps >= self.maxLearningSteps:
            return True
        return False
        
        
class ContinuousOptimizer(BlackBoxOptimizer):
    """ A more restricted class of black-box optimization algorithms
    that assume the parameters to be necessarily an array of continuous values 
    (which can be wrapped in a ParameterContainer). """    
        
    # dimension of the search space
    numParameters = None
    
    def _setInitEvaluable(self, evaluable):
        """ If the parameters are wrapped, we keep track of the wrapper explicitly. """
        if evaluable is None:
            # if there is no initial point specified, we start at one that's sampled 
            # normally around the origin.
            assert self.numParameters is not None
            evaluable = randn(self.numParameters)
            
        elif isinstance(evaluable, ParameterContainer):
            # in this case we have to wrap the evaluator
            self.wrappingEvaluable = evaluable.copy()
            self.wasWrapped = True
            evaluable = evaluable.params.copy()
        else:
            self.wasWrapped = False
        BlackBoxOptimizer._setInitEvaluable(self, evaluable)
        
    def _oneEvaluation(self, evaluable):        
        if self.wasWrapped:
            self.wrappingEvaluable._setParameters(evaluable)
            evaluable = self.wrappingEvaluable.copy()
        return BlackBoxOptimizer._oneEvaluation(self, evaluable)
        
