__author__ = 'Tom Schaul, tom@idsia.ch'

import threading
from scipy import ravel

from learning import LearningAgent
from pybrain.utilities import threaded


class OptimizationAgent(LearningAgent):
    """ A kind of learning agent that allows the use of any optimization
    algorithm for RL. """
    
    def __init__(self, module, learnerclass, **learnerArgs):
        self.newFitnessEvent = threading.Event()
        self.lastFitness = None
        LearningAgent.__init__(self, module)
        self.startLearner(learnerclass, module, **learnerArgs)
        
    @threaded()
    def startLearner(self, learnerclass, module, **learnerArgs):
        self.learner = learnerclass(self._episodeFitness, module, **learnerArgs)
        self.learner.maxEvaluations = None
        self.learner.maxLearningSteps = None
        self.module, _ = self.learner.learn()
        
    def _episodeFitness(self, module):
        self.module = module
        if self.lastFitness is None:
            #print 'Learner waiting...'
            self.newFitnessEvent.wait()        
        tmp = self.lastFitness
        self.lastFitness = None
        #print 'Learner finished waiting!'
        self.newFitnessEvent.set()
        self.newFitnessEvent.clear()
        return tmp
    
    def newEpisode(self):
        if len(self.history) > 0:
            if self.lastFitness is not None:     
                #print 'Agent waiting...'       
                self.newFitnessEvent.wait()
            #print 'Agent finished waiting!'
            self.lastFitness = self.summedRewards()
            self.history.clear()
            #print 'fit', self.lastFitness
            self.newFitnessEvent.set()
            self.newFitnessEvent.clear()
        LearningAgent.newEpisode(self)
        
    def summedRewards(self):
        s = 0.
        for seq in self.history:
            for _, _, reward in seq:
                s += ravel(reward)[0]
        return s
    