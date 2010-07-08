__author__ = 'Frank Sehnke, sehnke@in.tum.de'

#########################################################################
# The tasks availabele in the FlexCube Environment
#
# The FlexCube Environment is a Mass-Spring-System composed of 8 mass points.
# These resemble a cube with flexible edges.
#
# Tasks available are:
# - GrowTask, agent has to maximize the volume of the cube
# - JumpTask, agent has to maximize the distance of the lowest mass point to the floor
# - WalkTask, agent has to maximize the distance to the starting point
# - WalkDirectionTask, agent has to minimize the distance to a target point.
# - TargetTask, like the previous task but with several target points
#
#########################################################################


from pybrain.rl.environments import EpisodicTask
from scipy import array, r_, clip
import sensors

#Task basis class
class NoRewardTask(EpisodicTask):
    ''' just a basic task, that doesn't return a reward '''
    def __init__(self, env):
        EpisodicTask.__init__(self, env)
        self.step = 0
        self.epiStep = 0
        self.reward = [0.0]
        self.rawReward = 0.0
        self.obsSensors = ["EdgesReal"]
        self.rewardSensor = [""]
        self.oldReward = 0.0
        self.plotString = ["World Interactions", "Reward", "Reward on NoReward Task"]
        self.inDim = len(self.getObservation())
        self.outDim = self.env.actLen
        self.dif = (self.env.fraktMax - self.env.fraktMin) * self.env.dists[0]
        self.maxSpeed = self.dif / 30.0
        self.picCount = 0
        self.epiLen = 1

    def incStep(self):
        self.step += 1
        self.epiStep += 1

    def getReward(self):
        # calculate reward and return self.reward
        self.reward[0] = self.rawReward - self.getPain()
        return self.reward[0]

    def getObservation(self):
        # do something with self.sensors and return observation
        self.oldReward = self.rawReward
        aktSensors = self.env.getSensors()
        output = array([])
        for i in aktSensors:
            for j in self.obsSensors:
                if i[0] == j:
                    momSense = i[2]
                    output = r_[output, momSense]
            if i[0] == self.rewardSensor[0]:
                self.rawReward = i[2][0]
            if i[0] == "EdgesReal":
                self.EdgeL = momSense.copy()
        return output[:]

    #An agent can find easily the resonance frequency of the cube
    #Most tasks can be tricked by realising a resonance catastrophy
    #Therefore edges longer than 30 pixels result in punishment for all tasks
    def getPain(self):
        self.EdgeL = clip(self.EdgeL, 1.0, 4.0)
        return ((self.EdgeL - 1.0) ** 2).sum(axis=0)

    def performAction(self, action):
        """ a filtered mapping towards performAction of the underlying environment. """
        # scaling
        self.incStep()
        action = (action + 1.0) / 2.0 * self.dif + self.env.fraktMin * self.env.dists[0]
        #Clipping the maximal change in actions (max force clipping)
        action = clip(action, self.action - self.maxSpeed, self.action + self.maxSpeed)
        EpisodicTask.performAction(self, action)
        self.action = action.copy()

    def reset(self):
        self.reward[0] = 0.0
        self.rawReward = 0.0
        self.env.reset()
        self.action = [self.env.dists[0]] * self.outDim
        self.epiStep = 0
        EpisodicTask.reset(self)

    def isFinished(self):
        return (self.epiStep >= self.epiLen)

#Aim is to maximize the edge lengths (best reward: 3096.167, PGPE)
class GrowTask(NoRewardTask):
    def __init__(self, env):
        NoRewardTask.__init__(self, env)
        self.rewardSensor = ["EdgesSumReal"]
        self.obsSensors = ["EdgesReal", "EdgesTarget"]
        self.inDim = len(self.getObservation())
        self.plotString = ["World Interactions", "Size", "Reward on Growing Task"]
        self.env.mySensors = sensors.Sensors(self.obsSensors + self.rewardSensor)
        self.epiLen = 200 #suggested episode length

#Aim is to maximize the distance to the starting point  (best reward: 406.43, PGPE)
class WalkTask(NoRewardTask):
    def __init__(self, env):
        NoRewardTask.__init__(self, env)
        self.rewardSensor = ["DistToOrigin"]
        self.obsSensors = ["EdgesTarget", "EdgesReal", "VerticesContact", "Time"]
        self.inDim = len(self.getObservation())
        self.plotString = ["World Interactions", "Distance", "Reward on Walking Task"]
        self.env.mySensors = sensors.Sensors(self.obsSensors + self.rewardSensor)
        self.epiLen = 2000  #suggested episode length

    def getReward(self):
        if self.epiStep < self.epiLen: self.reward[0] = -self.getPain()
        else: self.reward[0] = self.rawReward * 800.0 - self.getPain()
        return self.reward[0]

#Aim is to maximize the distance to the starting point, but after a high fall (fun task)
class RollingUpTask(WalkTask):
    def __init__(self, env):
        WalkTask.__init__(self, env)
        self.env.startHight = 200.0
        self.env.reset()
        self.epiLen = 500  #suggested episode length

#Aim is to minimize distance to a target point
class WalkDirectionTask(WalkTask):
    def __init__(self, env):
        WalkTask.__init__(self, env)
        self.rewardSensor = ["Target"]
        self.obsSensors.append("Target")
        self.inDim = len(self.getObservation())
        self.plotString = ["World Interactions", "Distance", "Reward on Target Approach Task"]
        self.env.mySensors = sensors.Sensors(self.obsSensors)
        self.env.mySensors.sensors[4].targetList = [array([160.0, 0.0, 0.0])]
        if self.env.hasRenderInterface(): self.env.getRenderInterface().target = self.env.mySensors.sensors[4].targetList[0]
        self.epiLen = 2000
        #self.epiFakt=1.0/float(self.epiLen)

    def getReward(self):
        if self.epiStep < self.epiLen:
            if self.rawReward < 0.5: self.reward[0] = (0.5 - self.rawReward) * 1.0 - self.getPain()
            else: self.reward[0] = -self.getPain()
        else: self.reward[0] = clip(160.0 * (1.0 - self.rawReward), 0.0, 160.0) - self.getPain()
        return self.reward[0]

    def reset(self):
        WalkTask.reset(self)
        self.env.mySensors.sensors[4].targetList = [array([160.0, 0.0, 0.0])]
        if self.env.hasRenderInterface(): self.env.getRenderInterface().target = self.env.mySensors.sensors[4].targetList[0]

#Aim is to minimize distance to a variable target p<aoint
class TargetTask(WalkDirectionTask):
    def __init__(self, env):
        WalkDirectionTask.__init__(self, env)
        self.epiLen = 6000
        self.epiFakt = 1.0 / self.epiLen

    def getReward(self):
        if self.epiStep == self.epiLen / 3 or self.epiStep == 2 * self.epiLen / 3 or self.epiStep == self.epiLen:
            self.reward[0] = clip(160.0 * (1.0 - self.rawReward), 0.0, 160.0) - self.getPain()
        else: self.reward[0] = -self.getPain()
        return self.reward[0]

    def isFinished(self):
        if self.epiStep == self.epiLen / 3:
            self.env.reset()
            self.env.mySensors.sensors[4].targetList = [array([113.2, 0.0, -113.2])]
            if self.env.hasRenderInterface(): self.env.getRenderInterface().target = self.env.mySensors.sensors[4].targetList[0]
        if self.epiStep == 2 * self.epiLen / 3:
            self.env.reset()
            self.env.mySensors.sensors[4].targetList = [array([113.2, 0.0, 113.2])]
            if self.env.hasRenderInterface(): self.env.getRenderInterface().target = self.env.mySensors.sensors[4].targetList[0]
        return (self.epiStep >= self.epiLen)

#Aim is to maximize distance to floor (closest vertex counts)
class JumpTask(NoRewardTask):
    def __init__(self, env):
        NoRewardTask.__init__(self, env)
        self.rewardSensor = ["VerticesMinHight"]
        self.obsSensors = ["EdgesTarget", "EdgesReal", "VerticesContact"]
        self.inDim = len(self.getObservation())
        self.plotString = ["World Interactions", "Distance", "Reward on Walking Task"]
        self.env.mySensors = sensors.Sensors(self.obsSensors + self.rewardSensor)
        self.epiLen = 500
        self.maxReward = 0.0
        self.maxSpeed = self.dif / 10.0

    def getReward(self):
        if self.epiStep < self.epiLen:
            if self.rawReward > self.maxReward: self.maxReward = self.rawReward
            self.reward[0] = -self.getPain()
        else: self.reward[0] = self.maxReward - self.getPain()
        return self.reward[0]

    def reset(self):
        NoRewardTask.reset(self)
        self.maxReward = 0.0

