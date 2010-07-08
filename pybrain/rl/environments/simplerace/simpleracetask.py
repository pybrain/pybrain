__author__ = 'Julian Togelius, julian@idsia.ch'

from pybrain.rl.environments import EpisodicTask
from simpleracetcp import SimpleraceEnvironment

class SimpleraceTask(EpisodicTask):

    def getTotalReward(self):
        #score handled by environment?
        return self.environment.firstCarScore

    def getReward(self):
        return 0

    def isFinished(self):
        #this task can't really fail, a bad policy will just get a low score
        return False

    def setMaxLength(self, n):
        # I don't think this can be done
        pass

    def reset(self):
        EpisodicTask.reset(self)
        self.t = 0

    def performAction(self, action):
        self.t += 1
        EpisodicTask.performAction(self, action)

    def __init__(self):
        self.environment = SimpleraceEnvironment()
        EpisodicTask.__init__(self, self.environment)
        self.t = 0

