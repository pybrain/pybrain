from pybrain.rl.tasks import EpisodicTask
from pybrain.rl.environments.ode.sensors import *
from scipy import pi, ones, tanh, zeros, clip, array, random, sqrt
from time import sleep

#Basic class for all Johnnie tasks
class CCRLTask(EpisodicTask):
    def __init__(self, env):
        EpisodicTask.__init__(self, env)
        #Overall maximal tourque - is multiplied with relative max tourque for individual joint.
        self.maxPower=100.0
        self.reward_history = []
        self.count = 0 #timestep counter
        self.epiLen=1500 #suggestet episodic length for normal Johnnie tasks
        self.incLearn=0 #counts the task resets for incrementall learning
        self.env.FricMu=20.0 #We need higher friction for CCRL
        self.env.dt=0.002 #We also need more timly resolution

        # normalize standard sensors to (-1, 1)
        self.sensor_limits=[]
        #Angle sensors
        for i in range(self.env.actLen):
            self.sensor_limits.append((self.env.cLowList[i], self.env.cHighList[i]))            
        # Joint velocity sensors
        for i in range(self.env.actLen):
            self.sensor_limits.append((-20, 20))
        #Norm all actor dimensions to (-1, 1)
        self.actor_limits = [(-1, 1)]*env.actLen
        self.oldAction=zeros(env.actLen, float)
        self.dist=zeros(9, float)
        self.dif=array([0.0,0.0,0.0])
        self.target=array([-6.5,1.75,-10.0])
        self.tableFlag=0.0
        self.env.addSensor(SpecificBodyPositionSensor(['glas'], "glasPos"))
        self.env.addSensor(SpecificBodyPositionSensor(['palmLeft'], "palmPos"))
        self.env.addSensor(SpecificBodyPositionSensor(['fingerLeft1'], "finger1Pos"))
        self.env.addSensor(SpecificBodyPositionSensor(['fingerLeft2'], "finger2Pos"))
        #we changed sensors so we need to update environments sensorLength variable
        self.env.obsLen=len(self.env.getSensors())
        #normalization for the task spezific sensors
        for i in range(self.env.obsLen-2*self.env.actLen):
            self.sensor_limits.append((-4, 4))
        


    def getObservation(self):
        """ a filtered mapping to getSample of the underlying environment. """
        sensors = self.env.getSensors()
        #Sensor hand to target object
        for i in range(3):
            self.dist[i]=(sensors[self.env.obsLen-9+i]+sensors[self.env.obsLen-6+i]+sensors[self.env.obsLen-3+i])/3.0-(sensors[self.env.obsLen-12+i]+self.dif[i])
        #Sensor hand angle to horizontal plane X-Axis
        for i in range(3):
            self.dist[i+3]=(sensors[self.env.obsLen-3+i]-sensors[self.env.obsLen-6+i])*5.0
        #Sensor hand angle to horizontal plane Y-Axis
        for i in range(3):
            self.dist[i+6]=((sensors[self.env.obsLen-3+i]+sensors[self.env.obsLen-6+i])/2.0 - sensors[self.env.obsLen-9+i])*10.0
        if self.sensor_limits:
            sensors = self.normalize(sensors)
        sens=[]
        for i in range(self.env.obsLen-12):
            sens.append(sensors[i])
        for i in range(9):
            sens.append(self.dist[i])
        for i in self.oldAction:
            sens.append(i)
        return sens

    def performAction(self, action):
        self.oldAction=action
        #Filtered mapping towards performAction of the underlying environment   
        #The standard CCRL task uses a PID controller to controll directly angles instead of forces
        #This makes most tasks much simpler to learn             
        self.oldAction=action
        #Grasping as reflex depending on the distance to target - comment in for more easy grasping
        #if abs(self.dist[2])<0.5: action[15]=1.0
        #else: action[15]=-1.0
        isJoints=array(self.env.getSensorByName('JointSensor')) #The joint angles
        isSpeeds=array(self.env.getSensorByName('JointVelocitySensor')) #The joint angular velocitys
        act=(action+1.0)/2.0*(self.env.cHighList-self.env.cLowList)+self.env.cLowList #norm output to action intervall  
        action=tanh((act-isJoints-0.9*isSpeeds*self.env.tourqueList)*16.0)*self.maxPower*self.env.tourqueList #simple PID
        self.env.performAction(action)

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
        self.incLearn+=1
        self.reward_history.append(self.getTotalReward())
        self.tableFlag=0.0

    def getReward(self):
        if self.env.glasSum >= 2: grip=1.0 + float(self.env.glasSum-2)
        else: grip = 0.0
        if self.env.tableSum > 0: self.tableFlag=10.0
        dis=sqrt((self.dist**2).sum())
        nig=(abs(self.dist[4])+1.0)
        if self.env.stepCounter==self.epiLen:
            self.dist[3]=0.0
            self.dist[8]=0.0
            return 10.0-dis+grip/nig-self.tableFlag
        else:
            self.dist[3]=0.0
            self.dist[8]=0.0
            #print "dist:", self.dist[:3]
            return (10.0-dis)/float(self.epiLen)+(grip/nig-float(self.env.tableSum))*0.05
