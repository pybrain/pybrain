__author__ = 'Frank Sehnke, sehnke@in.tum.de'

from pybrain.rl.environments import EpisodicTask
from pybrain.rl.environments.ode.sensors import SpecificBodyPositionSensor
from scipy import tanh, zeros, array, random, sqrt, asarray


#Basic class for all ccrl tasks
class CCRLTask(EpisodicTask):
    def __init__(self, env):
        EpisodicTask.__init__(self, env)
        #Overall maximal tourque - is multiplied with relative max tourque for individual joint.
        self.maxPower = 100.0
        self.reward_history = []
        self.count = 0 #timestep counter
        self.epiLen = 1500 #suggestet episodic length for normal Johnnie tasks
        self.incLearn = 0 #counts the task resets for incrementall learning
        self.env.FricMu = 20.0 #We need higher friction for CCRL
        self.env.dt = 0.002 #We also need more timly resolution

        # normalize standard sensors to (-1, 1)
        self.sensor_limits = []
        #Angle sensors
        for i in range(self.env.actLen):
            self.sensor_limits.append((self.env.cLowList[i], self.env.cHighList[i]))
        # Joint velocity sensors
        for i in range(self.env.actLen):
            self.sensor_limits.append((-20, 20))
        #Norm all actor dimensions to (-1, 1)
        self.actor_limits = [(-1, 1)] * env.actLen
        self.oldAction = zeros(env.actLen, float)
        self.dist = zeros(9, float)
        self.dif = array([0.0, 0.0, 0.0])
        self.target = array([-6.5, 1.75, -10.5])
        self.grepRew = 0.0
        self.tableFlag = 0.0
        self.env.addSensor(SpecificBodyPositionSensor(['objectP00'], "glasPos"))
        self.env.addSensor(SpecificBodyPositionSensor(['palmLeft'], "palmPos"))
        self.env.addSensor(SpecificBodyPositionSensor(['fingerLeft1'], "finger1Pos"))
        self.env.addSensor(SpecificBodyPositionSensor(['fingerLeft2'], "finger2Pos"))
        #we changed sensors so we need to update environments sensorLength variable
        self.env.obsLen = len(self.env.getSensors())
        #normalization for the task spezific sensors
        for i in range(self.env.obsLen - 2 * self.env.actLen):
            self.sensor_limits.append((-4, 4))
        self.actor_limits = None

    def getObservation(self):
        """ a filtered mapping to getSample of the underlying environment. """
        sensors = self.env.getSensors()
        #Sensor hand to target object
        for i in range(3):
            self.dist[i] = ((sensors[self.env.obsLen - 9 + i] + sensors[self.env.obsLen - 6 + i] + sensors[self.env.obsLen - 3 + i]) / 3.0 - (sensors[self.env.obsLen - 12 + i] + self.dif[i])) * 4.0 #sensors[self.env.obsLen-12+i]
        #Sensor hand angle to horizontal plane X-Axis
        for i in range(3):
            self.dist[i + 3] = (sensors[self.env.obsLen - 3 + i] - sensors[self.env.obsLen - 6 + i]) * 5.0
        #Sensor hand angle to horizontal plane Y-Axis
        for i in range(3):
            self.dist[i + 6] = ((sensors[self.env.obsLen - 3 + i] + sensors[self.env.obsLen - 6 + i]) / 2.0 - sensors[self.env.obsLen - 9 + i]) * 10.0
        if self.sensor_limits:
            sensors = self.normalize(sensors)
        sens = []
        for i in range(self.env.obsLen - 12):
            sens.append(sensors[i])
        for i in range(9):
            sens.append(self.dist[i])
        for i in self.oldAction:
            sens.append(i)
        return sens

    def performAction(self, action):
        #Filtered mapping towards performAction of the underlying environment
        #The standard CCRL task uses a PID controller to controll directly angles instead of forces
        #This makes most tasks much simpler to learn
        self.oldAction = action
        #Grasping as reflex depending on the distance to target - comment in for more easy grasping
        if abs(abs(self.dist[:3]).sum())<2.0: action[15]=1.0 #self.grepRew=action[15]*.01
        else: action[15]=-1.0 #self.grepRew=action[15]*-.03
        isJoints=array(self.env.getSensorByName('JointSensor')) #The joint angles
        isSpeeds=array(self.env.getSensorByName('JointVelocitySensor')) #The joint angular velocitys
        act=(action+1.0)/2.0*(self.env.cHighList-self.env.cLowList)+self.env.cLowList #norm output to action intervall
        action=tanh((act-isJoints-0.9*isSpeeds*self.env.tourqueList)*16.0)*self.maxPower*self.env.tourqueList #simple PID
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
        self.tableFlag = 0.0

    def getReward(self):
        #rewarded for approaching the object
        dis = sqrt((self.dist[0:3] ** 2).sum())
        return (25.0 - dis) / float(self.epiLen) - float(self.env.tableSum) * 0.1

#Learn to grasp a glas at a fixed location
class CCRLGlasTask(CCRLTask):
    def __init__(self, env):
        CCRLTask.__init__(self, env)
        self.dif = array([0.0, 0.0, 0.0])
        self.epiLen = 1000 #suggestet episodic length for normal Johnnie tasks

    def isFinished(self):
        #returns true if episode timesteps has reached episode length and resets the task
        if self.count > self.epiLen:
            self.res()
            return True
        else:
            self.count += 1
            return False

    def getReward(self):
        if self.env.glasSum >= 2: grip = 1000.0
        else: grip = 0.0
        if self.env.tableSum > 0: self.tableFlag = -1.0
        else: tableFlag = 0.0
        self.dist[3] = 0.0
        self.dist[8] = 0.0
        dis = 100.0/((self.dist[:3] ** 2).sum()+0.1)
        nig = 10.0/((self.dist[3:] ** 2).sum()+0.1)
        if self.env.stepCounter == self.epiLen: print("Grip:", grip, "Dis:", dis, "Nig:", nig, "Table:", self.tableFlag)
        return (10 + grip + nig + dis + self.tableFlag) / float(self.epiLen) #-dis
        #else:
        #    return (25.0 - dis) / float(self.epiLen) + (grip / nig - float(self.env.tableSum)) * 0.1 #+self.grepRew (10.0-dis)/float(self.epiLen)+

#Learn to grasp a plate at a fixed location
class CCRLPlateTask(CCRLTask):
    def __init__(self, env):
        CCRLTask.__init__(self, env)
        self.dif = array([0.0, 0.2, 0.8])
        self.epiLen = 1000 #suggestet episodic length for normal Johnnie tasks

    def isFinished(self):
        #returns true if episode timesteps has reached episode length and resets the task
        if self.count > self.epiLen:
            self.res()
            return True
        else:
            if self.count == 1: self.pertGlasPos(0)
            self.count += 1
            return False

    def pertGlasPos(self, num):
        if num == 0: self.env.pert = asarray([0.0, 0.0, 0.5])

    def getReward(self):
        if self.env.glasSum >= 2: grip = 1.0
        else: grip = 0.0
        if self.env.tableSum > 0: self.tableFlag = 10.0
        #self.dist[4]=0.0
        #self.dist[8]=0.0
        dis = sqrt((self.dist[0:3] ** 2).sum())
        if self.count == self.epiLen:
            return 25.0 + grip - dis - self.tableFlag #/nig
        else:
            return (25.0 - dis) / float(self.epiLen) + (grip - float(self.env.tableSum)) * 0.1 #/nig -(1.0+self.oldAction[15])

#Learn to grasp a glas at 5 different locations
class CCRLGlasVarTask(CCRLGlasTask):
    def __init__(self, env):
        CCRLGlasTask.__init__(self, env)
        self.epiLen = 5000 #suggestet episodic length for normal Johnnie tasks

    def isFinished(self):
        #returns true if episode timesteps has reached episode length and resets the task
        if self.count > self.epiLen:
            self.res()
            return True
        else:
            if self.count == 1:
                self.pertGlasPos(0)
            if self.count == self.epiLen / 5 + 1:
                self.env.reset()
                self.pertGlasPos(1)
            if self.count == 2 * self.epiLen / 5 + 1:
                self.env.reset()
                self.pertGlasPos(2)
            if self.count == 3 * self.epiLen / 5 + 1:
                self.env.reset()
                self.pertGlasPos(3)
            if self.count == 4 * self.epiLen / 5 + 1:
                self.env.reset()
                self.pertGlasPos(4)
            self.count += 1
            return False

    def pertGlasPos(self, num):
        if num == 0: self.env.pert = asarray([1.0, 0.0, 0.5])
        if num == 1: self.env.pert = asarray([-1.0, 0.0, 0.5])
        if num == 2: self.env.pert = asarray([1.0, 0.0, 0.0])
        if num == 3: self.env.pert = asarray([-1.0, 0.0, 0.0])
        if num == 4: self.env.pert = asarray([0.0, 0.0, 0.25])

    def getReward(self):
        if self.env.glasSum >= 2: grip = 1.0
        else: grip = 0.0
        if self.env.tableSum > 0: self.tableFlag = 10.0
        self.dist[3] = 0.0
        self.dist[8] = 0.0
        dis = sqrt((self.dist ** 2).sum())
        nig = (abs(self.dist[4]) + 1.0)
        if self.count == self.epiLen or self.count == self.epiLen / 5 or self.count == 2 * self.epiLen / 5 or self.count == 3 * self.epiLen / 5 or self.count == 4 * self.epiLen / 5:
            return 25.0 + grip / nig - dis - self.tableFlag #/nig
        else:
            return (25.0 - dis) / float(self.epiLen) + (grip / nig - float(self.env.tableSum)) * 0.1 #/nig

#Learn to grasp a glas at random locations
class CCRLGlasVarRandTask(CCRLGlasVarTask):
    def pertGlasPos(self, num):
        self.env.pert = asarray([random.random()*2.0 - 1.0, 0.0, random.random()*0.5 + 0.5])


#Some experimental stuff
class CCRLPointTask(CCRLGlasVarTask):
    def __init__(self, env):
        CCRLGlasVarTask.__init__(self, env)
        self.epiLen = 1000 #suggestet episodic length for normal Johnnie tasks

    def isFinished(self):
        #returns true if episode timesteps has reached episode length and resets the task
        if self.count > self.epiLen:
            self.res()
            return True
        else:
            if self.count == 1:
                self.pertGlasPos(0)
            self.count += 1
            return False

    def getObservation(self):
        """ a filtered mapping to getSample of the underlying environment. """
        sensors = self.env.getSensors()
        sensSort = []
        #Angle and angleVelocity
        for i in range(32):
            sensSort.append(sensors[i])
        #Angles wanted (old action)
        for i in self.oldAction:
            sensSort.append(i)
        #Hand position
        for i in range(3):
            sensSort.append((sensors[38 + i] + sensors[41 + i]) / 2)
        #Hand orientation (Hack - make correkt!!!!)
        sensSort.append((sensors[38] - sensors[41]) / 2 - sensors[35]) #pitch
        sensSort.append((sensors[38 + 1] - sensors[41 + 1]) / 2 - sensors[35 + 1]) #yaw
        sensSort.append((sensors[38 + 1] - sensors[41 + 1])) #roll
        #Target position
        for i in range(3):
            sensSort.append(self.target[i])
        #Target orientation
        for i in range(3):
            sensSort.append(0.0)
        #Object type (start with random)
        sensSort.append(float(random.randint(-1, 1))) #roll

        #normalisation
        if self.sensor_limits:
            sensors = self.normalize(sensors)
        sens = []
        for i in range(32):
            sens.append(sensors[i])
        for i in range(29):
            sens.append(sensSort[i + 32])

        #calc dist to target
        self.dist = array([(sens[54] - sens[48]), (sens[55] - sens[49]), (sens[56] - sens[50]), sens[51], sens[52], sens[53], sens[15]])
        return sens

    def pertGlasPos(self, num):
        if num == 0: self.target = asarray([0.0, 0.0, 1.0])
        self.env.pert = self.target.copy()
        self.target = self.target.copy() + array([-6.5, 1.75, -10.5])

    def getReward(self):
        dis = sqrt((self.dist ** 2).sum())
        return (25.0 - dis) / float(self.epiLen) - float(self.env.tableSum) * 0.1

class CCRLPointVarTask(CCRLPointTask):
    def __init__(self, env):
        CCRLPointTask.__init__(self, env)
        self.epiLen = 2000 #suggestet episodic length for normal Johnnie tasks

    def isFinished(self):
        #returns true if episode timesteps has reached episode length and resets the task
        if self.count > self.epiLen:
            self.res()
            return True
        else:
            if self.count == 1:
                self.pertGlasPos(0)
            if self.count == self.epiLen / 2 + 1:
                self.env.reset()
                self.pertGlasPos(1)
            self.count += 1
            return False

    def getObservation(self):
        """ a filtered mapping to getSample of the underlying environment. """
        sensors = self.env.getSensors()
        sensSort = []
        #Angle and angleVelocity
        for i in range(32):
            sensSort.append(sensors[i])
        #Angles wanted (old action)
        for i in self.oldAction:
            sensSort.append(i)
        #Hand position
        for i in range(3):
            sensSort.append((sensors[38 + i] + sensors[41 + i]) / 2)
        #Hand orientation (Hack - make correkt!!!!)
        sensSort.append((sensors[38] - sensors[41]) / 2 - sensors[35]) #pitch
        sensSort.append((sensors[38 + 1] - sensors[41 + 1]) / 2 - sensors[35 + 1]) #yaw
        sensSort.append((sensors[38 + 1] - sensors[41 + 1])) #roll
        #Target position
        for i in range(3):
            sensSort.append(self.target[i])
        #Target orientation
        for i in range(3):
            sensSort.append(0.0)
        #Object type (start with random)
        sensSort.append(float(random.randint(-1, 1))) #roll

        #normalisation
        if self.sensor_limits:
            sensors = self.normalize(sensors)
        sens = []
        for i in range(32):
            sens.append(sensors[i])
        for i in range(29):
            sens.append(sensSort[i + 32])

        #calc dist to target
        self.dist = array([(sens[54] - sens[48]) * 10.0, (sens[55] - sens[49]) * 10.0, (sens[56] - sens[50]) * 10.0, sens[51], sens[52], sens[53], 1.0 + sens[15]])
        return sens

    def pertGlasPos(self, num):
        if num == 0: self.target = asarray([1.0, 0.0, 1.0])
        if num == 1: self.target = asarray([-1.0, 0.0, 1.0])
        if num == 2: self.target = asarray([1.0, 0.0, 0.0])
        if num == 3: self.target = asarray([-1.0, 0.0, 0.0])
        if num == 4: self.target = asarray([0.0, 0.0, 0.5])
        self.env.pert = self.target.copy()
        self.target = self.target.copy() + array([-6.5, 1.75, -10.5])

    def getReward(self):
        dis = sqrt((self.dist ** 2).sum())
        subEpi = self.epiLen / 2
        if self.count == self.epiLen or self.count == subEpi:
            return (25.0 - dis) / 2.0
        else:
            return (25.0 - dis) / float(self.epiLen) - float(self.env.tableSum) * 0.1

