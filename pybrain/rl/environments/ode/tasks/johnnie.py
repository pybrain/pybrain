__author__ = 'Frank Sehnke, sehnke@in.tum.de'

from pybrain.rl.environments import EpisodicTask
from pybrain.rl.environments.ode.sensors import * #@UnusedWildImport
from scipy import  ones, tanh, clip

#Basic class for all Johnnie tasks
class JohnnieTask(EpisodicTask):
    def __init__(self, env):
        EpisodicTask.__init__(self, env)
        self.maxPower = 100.0 #Overall maximal tourque - is multiplied with relative max tourque for individual joint to get individual max tourque
        self.reward_history = []
        self.count = 0 #timestep counter
        self.epiLen = 500 #suggestet episodic length for normal Johnnie tasks
        self.incLearn = 0 #counts the task resets for incrementall learning
        self.env.FricMu = 20.0 #We need higher friction for Johnnie
        self.env.dt = 0.01 #We also need more timly resolution

        # normalize standard sensors to (-1, 1)
        self.sensor_limits = []
        #Angle sensors
        for i in range(self.env.actLen):
            self.sensor_limits.append((self.env.cLowList[i], self.env.cHighList[i]))
        # Joint velocity sensors
        for i in range(self.env.actLen):
            self.sensor_limits.append((-20, 20))
        #Norm all actor dimensions to (-1, 1)
        #self.actor_limits = [(-1, 1)] * env.actLen
        self.actor_limits = None

    def performAction(self, action):
        #Filtered mapping towards performAction of the underlying environment
        #The standard Johnnie task uses a PID controller to controll directly angles instead of forces
        #This makes most tasks much simpler to learn
        isJoints=self.env.getSensorByName('JointSensor') #The joint angles
        isSpeeds=self.env.getSensorByName('JointVelocitySensor') #The joint angular velocitys
        act=(action+1.0)/2.0*(self.env.cHighList-self.env.cLowList)+self.env.cLowList #norm output to action intervall
        action=tanh((act-isJoints-isSpeeds)*16.0)*self.maxPower*self.env.tourqueList #simple PID
        EpisodicTask.performAction(self, action)
        #self.env.performAction(action)

    def isFinished(self):
        #returns true if episode timesteps has reached episode length and resets the task
        if self.count > self.epiLen:
            self.res()
            return True
        else:
            self.count += 1
            return False

    def res(self):
        #sets counter and history back, increases incremental counter
        self.count = 0
        self.incLearn += 1
        self.reward_history.append(self.getTotalReward())

#The standing tasks, just not falling on its own is the goal
class StandingTask(JohnnieTask):
    def __init__(self, env):
        JohnnieTask.__init__(self, env)
        #add task spezific sensors, TODO build attitude sensors
        self.env.addSensor(SpecificBodyPositionSensor(['footLeft'], "footLPos"))
        self.env.addSensor(SpecificBodyPositionSensor(['footRight'], "footRPos"))
        self.env.addSensor(SpecificBodyPositionSensor(['palm'], "bodyPos"))
        self.env.addSensor(SpecificBodyPositionSensor(['head'], "headPos"))

        #we changed sensors so we need to update environments sensorLength variable
        self.env.obsLen = len(self.env.getSensors())

        #normalization for the task spezific sensors
        for _ in range(self.env.obsLen - 2 * self.env.actLen):
            self.sensor_limits.append((-20, 20))
        self.epiLen = 1000 #suggested episode length for this task

    def getReward(self):
        # calculate reward and return reward
        reward = self.env.getSensorByName('headPos')[1] / float(self.epiLen) #reward is hight of head
        #to prevent jumping reward can't get bigger than head position while standing absolut upright
        reward = clip(reward, -14.0, 4.0)
        return reward

#Robust standing task suited for complete learning with already standable controller
class RStandingTask(StandingTask):
    def __init__(self, env):
        StandingTask.__init__(self, env)
        self.epiLen = 4000 #suggested episode length for this task
        self.h1 = self.epiLen / 4 #timestep of first perturbation
        self.h2 = self.epiLen / 2 #timestep of environment reset
        self.h3 = 3 * self.epiLen / 4 #timestep of second perturbation
        self.pVect1 = (0, -9.81, -9.81) #gravity vector for first perturbation
        self.pVect2 = (0, -9.81, 0) #gravity vector standard
        self.pVect3 = (0, -9.81, 9.81) #gravity vector for second perturbation

    def isFinished(self):
        if self.count > self.epiLen:
            self.res()
            return True
        else:
            self.count += 1
            self.disturb()
            return False

    #changes gravity vector for perturbation
    def disturb(self):
        disturb = self.getDisturb()
        if self.count == self.h1: self.env.world.setGravity(self.pVect1)
        if self.count == self.h1 + disturb: self.env.world.setGravity(self.pVect2)
        if self.count == self.h2: self.env.reset()
        if self.count == self.h3: self.env.world.setGravity(self.pVect3)
        if self.count == self.h3 + disturb: self.env.world.setGravity(self.pVect2)

    def getDisturb(self):
        return 50

#Robust standing task suited for incremental learning with already standable controller
class RobStandingTask(RStandingTask):
    #increases the amount of perturbation with the number of episodes
    def getDisturb(self):
        return clip((10 + self.incLearn / 50), 0.0, 50)

#Robust standing task suited for complete learning with an empty controller
class RobustStandingTask(RobStandingTask):
    #increases the amount of perturbation with the number of episodes
    def getDisturb(self):
        return clip((self.incLearn / 200), 0.0, 50)

#The jumping tasks, goal is to maximize the highest point the head reaches during episode
class JumpingTask(JohnnieTask):
    def __init__(self, env):
        JohnnieTask.__init__(self, env)
        #add task spezific sensors, TODO build attitude sensors
        self.env.addSensor(SpecificBodyPositionSensor(['footLeft']))
        self.env.addSensor(SpecificBodyPositionSensor(['footRight']))
        self.env.addSensor(SpecificBodyPositionSensor(['palm']))
        self.env.addSensor(SpecificBodyPositionSensor(['head']))

        #we changed sensors so we need to update environments sensorLength variable
        self.env.obsLen = len(self.env.getSensors())

        #normalization for the task spezific sensors
        for _ in range(self.env.obsLen - 2 * self.env.actLen):
            self.sensor_limits.append((-20, 20))
        self.epiLen = 400 #suggested episode length for this task
        self.maxHight = 4.0 #maximum hight reached during episode
        self.maxPower = 400.0 #jumping needs more power

    def getReward(self):
        # calculate reward and return reward
        reward = self.env.getSensorByName('SpecificBodyPositionSensor8')[1] #reward is hight of head
        if reward > self.maxHight:
            self.maxHight = reward
        if self.count == self.epiLen:
            reward = self.maxHight
        else:
            reward = 0.0
        return reward

    def res(self):
        self.count = 0
        self.reward_history.append(self.getTotalReward())
        self.maxHight = 4.0

#The standing up from prone task, goal is to stand up from prone in an upright position.
#Nearly unsolveable task - most learners achive to bring Johnnie in some kind of kneeling position
class StandingUpTask(StandingTask):
    def __init__(self, env):
        StandingTask.__init__(self, env)
        self.epiLen = 2000 #suggested episode length for this task
        self.env.tourqueList[0] = 2.5
        self.env.tourqueList[1] = 2.5

    def getReward(self):
        # calculate reward and return reward
        if self.count < 800:
            return 0.0
        else:
            reward = self.env.getSensorByName('SpecificBodyPositionSensor8')[1] / float(self.epiLen - 800) #reward is hight of head
            #to prevent jumping reward can't get bigger than head position while standing absolut upright
            reward = clip(reward, -14.0, 4.0)
            return reward

    def performAction(self, action):
        if self.count < 800:
            #provoke falling
            a = ones(self.env.actLen, int) * self.maxPower * self.env.tourqueList * -1
            StandingTask.performAction(self, a)
        else:
            StandingTask.performAction(self, action)

