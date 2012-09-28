__author__ = 'Julian Togelius, julian@idsia.ch'

from scipy import array

from pybrain.rl.agents.agent import Agent


class SimpleController(Agent):

    def integrateObservation(self, obs):
        self.speed = obs[0]
        self.angleToCurrentWP = obs[1]
        self.distanceToCurrentWP = obs[2]
        self.angleToNextWP = obs[3]
        self.distanceToNextWP = obs[4]
        self.angleToOtherVehicle = obs[5]
        self.distanceToOtherVehicle = obs[6]


    def getAction(self):
        if self.speed < 10:
            driving = 1
        else:
            driving = 0
        if self.angleToCurrentWP > 0:
            steering = -1
        else:
            steering = 1
        print("speed", self.speed, "angle", self.angleToCurrentWP, "driving", driving, "steering", steering)
        return array([driving, steering])

