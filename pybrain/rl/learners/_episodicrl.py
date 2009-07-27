__author__ = 'Tom Schaul, tom@idsia.ch'

from learner import Learner
from policygradients import ENAC
from pybrain.rl.tasks.episodic import EpisodicTask
from pybrain.structure.modules.module import Module
from pybrain.rl.agents.policygradient import PolicyGradientAgent


class EpisodicRL(Learner):
    """ Learner interface for RL, for the simple episodic case with the agent just being a module. 
    Default settings:"""
    
    sublearner = ENAC
    subagent = PolicyGradientAgent
    subargs = {}
    learningBatches = 10
    learningRate = 0.05
    momentum = 0.9
    
    def __init__(self, evaluator, evaluable, **args):
        """ The evaluator must be an episodic task, an the evaluable must be a module. """
        assert isinstance(evaluator, EpisodicTask)
        assert isinstance(evaluable, Module)
        self.agent = self.subagent(evaluable.copy(), self.sublearner(**self.subargs))
        self.agent.learner.setAlpha(self.learningRate)
        self.agent.learner.gd.momentum = self.momentum
        self.agent.copy = lambda: self.agent
        self.module = evaluable.copy()
        def wrappedEvaluator(module):
            """ evaluate the internal agent (and changing its internal parameters),
            and then transferring the parameters to the outside basenet """
            self.agent.reset()
            res = 0.
            for dummy in range(self.learningBatches):
                res += evaluator(self.agent)
            res /= self.learningBatches
            self.agent.learn()
            module._setParameters(self.agent.module.params[:module.paramdim])
            # the performance is measured on a greedy run:
            res2 = evaluator(module)
            if self.verbose:
                print 'stoch', res, 'greedy', res2,
            return res2
        
        Learner.__init__(self, wrappedEvaluator, evaluable, **args)
        
    def _learnStep(self):
        if self.steps % (self.learningBatches+1) != 0:
            return
        fitness = self.evaluator(self.module)
        if fitness > self.bestEvaluation:
            self.bestEvaluable = self.module.copy()
            self.bestEvaluation = fitness
        if self.verbose:
            print self.steps, ':', self.bestEvaluation
        