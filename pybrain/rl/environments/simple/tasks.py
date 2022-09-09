__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.rl.environments import EpisodicTask


class MinimizeTask(EpisodicTask):
    def __init__(self, environment):
        EpisodicTask.__init__(self, environment)
        self.N = 15
        self.t = 0
        self.state = [0.0] * environment.dim
        self.action = [0.0] * environment.dim

    def reset(self):
        EpisodicTask.reset(self)
        self.t = 0

    def isFinished(self):
        if self.t >= self.N:
            self.t = 0
            return True
        else:
            self.t += 1
            return False

    def getObservation(self):
        self.state = EpisodicTask.getObservation(self)
        return self.state

    def performAction(self, action):
        EpisodicTask.performAction(self, action)
        self.action = action

    def getReward(self):
        # sleep(0.01)
        # print(self.state, self.action)
        reward = self.env.f([s + 0.1 * a for s, a in zip(self.state, self.action)])
        return - sum(reward)

