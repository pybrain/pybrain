__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.rl.environments import EpisodicTask
from scipy import pi

class GradualRewardTask(EpisodicTask):
    ''' task gives more reward, the higher the bar is.'''
    def __init__(self, environment):
        EpisodicTask.__init__(self, environment)
        self.maxPower = 0.5
        self.reward_history = []
        self.count = 0
        # normalize to (-1, 1)
        self.sensor_limits = [(-pi, pi), (-20, 20)]
        #self.actor_limits = [(-1, 1)]
        self.actor_limits = None

    def isFinished(self):
        if self.count > 1000:
            self.count = 0
            self.reward_history.append(self.getTotalReward())
            return True
        else:
            self.count += 1
            return False

    def getReward(self):
        # calculate reward and return reward
        jointSense = self.env.getSensorByName('JointSensor')
        veloSense = self.env.getSensorByName('JointVelocitySensor')

        j = jointSense[0]
        v = veloSense[0]

        reward = (abs(j)) ** 2 - 0.2 * abs(v)
        # time.sleep(0.001)
        return reward

    def performAction(self, action):
        EpisodicTask.performAction(self, action*self.maxPower)
