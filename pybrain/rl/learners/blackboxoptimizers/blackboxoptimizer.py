__author__ = 'Tom Schaul, tom@idsia.ch'


from scipy import ndarray, size, rand

from pybrain.rl.learners import Learner
from pybrain.structure.parametercontainer import ParameterContainer


class BlackBoxOptimizer(Learner):
    """ A type of Learner that optimizes an unknown (single-output) function.
    It only accepts evaluables that are arrays, or have a .params attribute which is an array. 
    
    Subclasses can implement a ._batchLearn() method instead of the _learnStep() method, 
    which will be called instead if the 'online' parameter is set accordingly. 
    """
    
    # minimize or maximize? 
    minimize = False
    
    # maximal number of function evaluations
    maxEvaluations = 1e6
    
    # stopping criterion
    stopPrecision = 1e-10   
    
    wrappingEvaluable = None
    
    noisyEvaluator = False
    
    # if None, the provided weights are kept. This is a list of tuples.
    initialSearchRange = None 
    
    online = True
    
    def __init__(self, evaluator, evaluable, **args):
        Learner.__init__(self, evaluator, evaluable, **args)
        
        if isinstance(evaluable, ParameterContainer):
            # in this case we have to wrap the evaluator
            self.wrappingEvaluable = evaluable.copy()
            self.wrappingEvaluable.name = 'opt-by-'+self.__class__.__name__
            def wrappedEvaluator(x):
                self.wrappingEvaluable._setParameters(x)
                return evaluator(self.wrappingEvaluable)
            self.evaluator = wrappedEvaluator
            self.x0 = evaluable.params.copy()
            self.bestEvaluable = self.x0.copy()
        else:
            self.x0 = evaluable.copy()
            
        if self.minimize:
            # then we need to change the sign of the evaluations
            tmp = self.evaluator
            self.evaluator = lambda x: -tmp(x)
            if self.desiredEvaluation != None:
                self.desiredEvaluation *= -1
            
        # the first guess at the solution (it must be an array)
        assert type(self.x0) == ndarray
        self.xdim = size(self.x0)
        if self.initialSearchRange != None:
            for i  in range(self.xdim):
                mi, ma = self.initialSearchRange[i]
                self.x0[i] = rand()*(ma-mi)+mi
        self.noisyEvaluator = evaluator.noisy
        
    def learn(self, maxSteps = None):
        """ Some BlackBoxOptimizers can only be called one time, and currently do not support iteratively
        adding more steps. """
        
        if not self.online:
            if self.maxEvaluations != None:
                if maxSteps != None:
                    maxSteps = min(maxSteps, self.maxEvaluations-self.steps)
                else:
                    maxSteps = self.maxEvaluations-self.steps
            self._batchLearn(maxSteps)
        else:
            Learner.learn(self, maxSteps)
        
        if self.wrappingEvaluable != None and isinstance(self.bestEvaluable, ndarray):
            xopt = self.bestEvaluable
            self.wrappingEvaluable._setParameters(xopt)
            self.bestEvaluable = self.wrappingEvaluable
        
        if self.minimize:
            self.bestEvaluation *= -1
        return self.bestEvaluable, self.bestEvaluation
        
    