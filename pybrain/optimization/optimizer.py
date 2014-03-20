__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import array, randn, ndarray, isinf, isnan, isscalar
import logging

from pybrain.utilities import setAllArgs, abstractMethod, DivergenceError
from pybrain.rl.learners.directsearch.directsearch import DirectSearchLearner
from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.rl.environments.functions.function import FunctionEnvironment
from pybrain.rl.environments.fitnessevaluator import FitnessEvaluator
from pybrain.rl.environments.functions.transformations import oppositeFunction
from pybrain.structure.evolvables.maskedmodule import MaskedModule
from pybrain.structure.evolvables.maskedparameters import MaskedParameters
from pybrain.structure.evolvables.topology import TopologyEvolvable
from pybrain.structure.modules.module import Module


class BlackBoxOptimizer(DirectSearchLearner):
    """ The super-class for learning algorithms that treat the problem as a black box. 
    At each step they change the policy, and get a fitness value by invoking 
    the FitnessEvaluator (provided as first argument upon initialization).
    
    Evaluable objects can be lists or arrays of continuous values (also wrapped in ParameterContainer) 
    or subclasses of Evolvable (that define its methods).
    """    
    
    
    # some algorithms are designed for minimization only, those can put this flag:
    mustMinimize = False
    mustMaximize = False
        
    #: Is there a known value of sufficient fitness?
    desiredEvaluation = None    

    #: Stopping criterion based on number of evaluations.    
    maxEvaluations = 1e6 
        
    #: Stopping criterion based on number of learning steps. 
    maxLearningSteps = None    
    
    
    #: dimension of the search space, if applicable
    numParameters = None
    
    '''added by JPQ Boundaries of the search space, if applicable'''
    xBound = None
    feasible = None
    constrained = None
    violation = None
    # ---   
    
    #: Store all evaluations (in the ._allEvaluations list)?
    storeAllEvaluations = False
    #: Store all evaluated instances (in the ._allEvaluated list)?
    storeAllEvaluated = False
    
    # an optimizer can take different forms of evaluables, and depending on its
    # needs, wrap them into a ParameterContainer (which is also an Evolvable)
    # or unwrap them to act directly on the array of parameters (all ContinuousOptimizers)
    _wasWrapped = False
    _wasUnwrapped = False
    _wasOpposed = False
        
    listener = None
    
    #: provide console output during learning
    verbose = False
    
    # some algorithms have a predetermined (minimal) number of 
    # evaluations they will perform during each learningStep:
    batchSize = 1
    
    
    def __init__(self, evaluator = None, initEvaluable = None, **kwargs):
        """ The evaluator is any callable object (e.g. a lambda function). 
        Algorithm parameters can be set here if provided as keyword arguments. """
        # set all algorithm-specific parameters in one go:
        self.__minimize = None
        self.__evaluator = None
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
            
    def _getMinimize(self): return self.__minimize
            
    def _setMinimize(self, flag):
        """ Minimization vs. maximization: priority to algorithm requirements, 
        then evaluator, default = maximize."""
        self.__minimize = flag
        opp = False
        if flag is True:
            if self.mustMaximize:
                opp = True
                self.__minimize = False
        if flag is False:
            if self.mustMinimize:
                opp = True
                self.__minimize = True       
        if self.__evaluator is not None:
            if opp is not self._wasOpposed: 
                self._flipDirection()
        self._wasOpposed = opp
        
    #: Minimize cost or maximize fitness? By default, all functions are maximized.    
    minimize = property(_getMinimize, _setMinimize)
        
    def setEvaluator(self, evaluator, initEvaluable = None):
        """ If not provided upon construction, the objective function can be given through this method.
        If necessary, also provide an initial evaluable."""
        
        # default settings, if provided by the evaluator:
        if isinstance(evaluator, FitnessEvaluator):
            if self.desiredEvaluation is None:
                self.desiredEvaluation = evaluator.desiredValue               
            if self.minimize is None:
                self.minimize = evaluator.toBeMinimized 
            # in some cases, we can deduce the dimension from the provided evaluator:
            if isinstance(evaluator, FunctionEnvironment):
                if self.numParameters is None:            
                    self.numParameters = evaluator.xdim
                elif self.numParameters is not evaluator.xdim:
                    raise ValueError("Parameter dimension mismatch: evaluator expects "+str(evaluator.xdim)\
                                     +" but it was set to "+str(self.numParameters)+".")
                '''added by JPQ to handle boundaries on the parameters'''
                self.evaluator = evaluator
                if self.xBound is None:            
                    self.xBound = evaluator.xbound
                if self.feasible is None:
                    self.feasible = evaluator.feasible
                if self.constrained is None:
                    self.constrained = evaluator.constrained
                if self.violation is None:
                    self.violation = evaluator.violation
                # ---
        # default: maximize
        if self.minimize is None:
            self.minimize = False
        self.__evaluator = evaluator
        if self._wasOpposed:
            self._flipDirection()
        #set the starting point for optimization (as provided, or randomly)
        self._setInitEvaluable(initEvaluable)        
        self.bestEvaluation = None
        self._additionalInit()
        self.bestEvaluable = self._initEvaluable
        
    def _flipDirection(self):
        self.__evaluator = oppositeFunction(self.__evaluator)
        if self.desiredEvaluation is not None:
            self.desiredEvaluation *= -1        
        
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
            self._wasWrapped = True
            evaluable = pc
        self._initEvaluable = evaluable
        if isinstance(self._initEvaluable, ParameterContainer):
            if self.numParameters is None:            
                self.numParameters = len(self._initEvaluable)
            elif self.numParameters != len(self._initEvaluable):
                raise ValueError("Parameter dimension mismatch: evaluator expects "+str(self.numParameters)\
                                 +" but the evaluable has "+str(len(self._initEvaluable))+".")
                  
    
    def learn(self, additionalLearningSteps = None):
        """ The main loop that does the learning. """
        assert self.__evaluator is not None, "No evaluator has been set. Learning cannot start."
        if additionalLearningSteps is not None:
            self.maxLearningSteps = self.numLearningSteps + additionalLearningSteps - 1
        while not self._stoppingCriterion():
            try:
                self._learnStep()
                self._notify()
                self.numLearningSteps += 1
            except DivergenceError:
                logging.warning("Algorithm diverged. Stopped after "+str(self.numLearningSteps)+" learning steps.")
                break
            except ValueError:
                logging.warning("Something numerical went wrong. Stopped after "+str(self.numLearningSteps)+" learning steps.")
                break
        return self._bestFound()
        
    def _learnStep(self):
        """ The core method to be implemented by all subclasses. """
        abstractMethod()        
        
    def _bestFound(self):
        """ return the best found evaluable and its associated fitness. """
        bestE = self.bestEvaluable.params.copy() if self._wasWrapped else self.bestEvaluable
        if self._wasOpposed and isscalar(self.bestEvaluation):
            bestF = -self.bestEvaluation
        else:
            bestF = self.bestEvaluation
        return bestE, bestF
        
    def _oneEvaluation(self, evaluable):
        """ This method should be called by all optimizers for producing an evaluation. """
        if self._wasUnwrapped:
            self.wrappingEvaluable._setParameters(evaluable)
            res = self.__evaluator(self.wrappingEvaluable)
        elif self._wasWrapped:            
            res = self.__evaluator(evaluable.params)
        else:            
            res = self.__evaluator(evaluable)
            ''' added by JPQ '''
            if self.constrained :
                self.feasible = self.__evaluator.outfeasible
                self.violation = self.__evaluator.outviolation
            # ---
        if isscalar(res):
            # detect numerical instability
            if isnan(res) or isinf(res):
                raise DivergenceError
            # always keep track of the best
            if (self.numEvaluations == 0
                or self.bestEvaluation is None
                or (self.minimize and res <= self.bestEvaluation)
                or (not self.minimize and res >= self.bestEvaluation)):
                self.bestEvaluation = res
                self.bestEvaluable = evaluable.copy()
        
        self.numEvaluations += 1
        
        # if desired, also keep track of all evaluables and/or their fitness.                        
        if self.storeAllEvaluated:
            if self._wasUnwrapped:            
                self._allEvaluated.append(self.wrappingEvaluable.copy())
            elif self._wasWrapped:            
                self._allEvaluated.append(evaluable.params.copy())
            else:            
                self._allEvaluated.append(evaluable.copy())        
        if self.storeAllEvaluations:
            if self._wasOpposed and isscalar(res):
                ''' added by JPQ '''
                if self.constrained :
                    self._allEvaluations.append([-res,self.feasible,self.violation])
                # ---
                else:
                    self._allEvaluations.append(-res)
            else:
                ''' added by JPQ '''
                if self.constrained :
                    self._allEvaluations.append([res,self.feasible,self.violation])
                # ---
                else:
                    self._allEvaluations.append(res)
        ''' added by JPQ '''
        if self.constrained :
            return [res,self.feasible,self.violation]
        else:
        # ---
            return res
    
    def _stoppingCriterion(self):
        if self.maxEvaluations is not None and self.numEvaluations+self.batchSize > self.maxEvaluations:
            return True
        if self.desiredEvaluation is not None and self.bestEvaluation is not None and isscalar(self.bestEvaluation):
            if ((self.minimize and self.bestEvaluation <= self.desiredEvaluation)
                or (not self.minimize and self.bestEvaluation >= self.desiredEvaluation)):
                return True
        if self.maxLearningSteps is not None and self.numLearningSteps > self.maxLearningSteps:
            return True
        return False
    
    def _notify(self):
        """ Provide some feedback during the run. """
        if self.verbose:
            print('Step:', self.numLearningSteps, 'best:', self.bestEvaluation)
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
            self._wasUnwrapped = True
        elif not (evaluable is None 
                  or isinstance(evaluable, list) 
                  or isinstance(evaluable, ndarray)):
            raise ValueError('Continuous optimization algorithms require a list, array or'+\
                             ' ParameterContainer as evaluable.')
        BlackBoxOptimizer._setInitEvaluable(self, evaluable)
        self._wasWrapped = False
        self._initEvaluable = self._initEvaluable.params.copy()     
        
    def _bestFound(self):
        """ return the best found evaluable and its associated fitness. """
        bestE, bestF = BlackBoxOptimizer._bestFound(self)
        if self._wasUnwrapped:
            self.wrappingEvaluable._setParameters(bestE)
            bestE = self.wrappingEvaluable.copy()
        return bestE, bestF
   

class TopologyOptimizer(BlackBoxOptimizer):
    """ A class of algorithms that changes the topology as well as the parameters.
    It does not accept an arbitrary Evolvable as initial point, only a 
    ParameterContainer (or a continuous vector). """
        
    def _setInitEvaluable(self, evaluable):
        BlackBoxOptimizer._setInitEvaluable(self, evaluable)
        # distinguish modules from parameter containers.
        if not isinstance(evaluable, TopologyEvolvable):
            if isinstance(evaluable, Module):
                self._initEvaluable = MaskedModule(self._initEvaluable)
            else:
                self._initEvaluable = MaskedParameters(self._initEvaluable, returnZeros = True)   
    
     
