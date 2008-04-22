__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.utilities import abstractMethod, Named


class Learner(Named):
    """ Takes as input an evaluator (i.e. that implements a .__call__() method), 
    and an object that can be evaluated with that. The result of an evaluation is a real number,
    which the Learner attempts to MAXIMIZE. 
    It tries to achieve this by changing the internals of the evaluable.
    An evaluable object must have a .copy() method.
    It could be an instance of e.g. numpy.array, ParameterContainer or Evolvable.
    """
    
    desiredEvaluation = None
    maxEvaluations = None
    
    # global flag for enabling plenty of information being written during the learning
    verbose = False
    
    
    def __init__(self, evaluator, evaluable, **args):
        self.setArgs(**args)
        self.evaluator = evaluator
        self.bestEvaluable = evaluable.copy()
        self.bestEvaluation = self.evaluator(self.bestEvaluable)
        if hasattr(evaluator, 'desiredValue'):
            self.desiredEvaluation = evaluator.desiredValue
        self.steps = 1
        
    def learn(self, maxSteps = None):
        """ @return: (best evaluable found, best fitness) """
        if maxSteps != None:
            maxSteps += self.steps
        while True:
            if maxSteps != None and self.steps >= maxSteps:
                break
            if self.maxEvaluations != None and self.steps >= self.maxEvaluations:
                break
            if self.desiredEvaluation != None and self.bestEvaluation >= self.desiredEvaluation:
                break
            if self._stoppingCriterion():
                break
            self._learnStep()
            self.steps += 1
        return self.bestEvaluable, self.bestEvaluation
    
    def _learnStep(self):
        """ do as much learning as possible, but while only doing a single evaluation. 
        This method should update self.bestEvaluable and self.bestEvaluation """
        abstractMethod()
        
    def _stoppingCriterion(self):
        """ subclasses can specify particular stopping criteria """
        return False