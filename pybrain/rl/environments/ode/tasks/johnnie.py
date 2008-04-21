from pybrain.rl.tasks import EpisodicTask
from pybrain.rl.environments.ode.sensors import *
from scipy import pi, ones, tanh, zeros

class JohnnieTask(EpisodicTask):
    """Basic class for all Johnnie tasks"""
    def __init__(self, env):
        EpisodicTask.__init__(self, env)
        self.maxPower=100.0 #Overall maximal tourque - is multiplied with relative max tourque for individual joint to get individual max tourque
        self.reward_history = []
        self.count = 0 #timestep counter
        self.epiLen=500 #suggestet episodic length for normal Johnnie tasks

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

    def performAction(self, action):
        """ a filtered mapping towards performAction of the underlying environment. """   
        #The standard Johnnie task uses a PID controller to controll directly angles instead of forces
        #This makes most tasks much simpler to learn             
        isJoints=self.env.getSensorByName('JointSensor') 
        isSpeeds=self.env.getSensorByName('JointVelocitySensor')
        act=(action+1.0)/2.0*(self.env.cHighList-self.env.cLowList)+self.env.cLowList #norm output to action intervall  
        action=tanh((act-isJoints-isSpeeds)*16.0)*self.maxPower*self.env.tourqueList #simple PID
        self.env.performAction(action)

    def isFinished(self):
        if self.count > self.epiLen:
            self.res()
            return True
        else:
            self.count += 1
            return False

    def res(self):
        self.count = 0 
        self.reward_history.append(self.getTotalReward())
            

class StandingTask(JohnnieTask):
    def __init__(self, env):
        JohnnieTask.__init__(self, env)
        #add task spezific sensors, TODO build attitude sensors
        self.env.addSensor(SpecificBodyPositionSensor(['footLeft'], "footLPos"))
        self.env.addSensor(SpecificBodyPositionSensor(['footRight'], "footRPos"))
        self.env.addSensor(SpecificBodyPositionSensor(['palm'], "bodyPos"))
        self.env.addSensor(SpecificBodyPositionSensor(['head'], "headPos"))

        #we changed sensors so we need to update environments sensorLength variable
        self.env.obsLen=len(self.env.getSensors())

        #normalization for the task spezifish sensors
        for i in range(self.env.obsLen-2*self.env.actLen):
            self.sensor_limits.append((-20, 20))
        self.epiLen=1000 #suggested episode length for this task        
        
    def getReward(self):
        # calculate reward and return reward
        reward=self.env.getSensorByName('headPos')[1]/float(self.epiLen) #reward is hight of head
        if reward>4.0: reward=4.0 #to prevent jumping reward can't get bigger than head position while standing absolut upright
        return reward

class JumpingTask(JohnnieTask):
    def __init__(self, env):
        JohnnieTask.__init__(self, env)
        #add task spezific sensors, TODO build attitude sensors
        self.env.addSensor(SpecificBodyPositionSensor(['footLeft']))
        self.env.addSensor(SpecificBodyPositionSensor(['footRight']))
        self.env.addSensor(SpecificBodyPositionSensor(['palm']))
        self.env.addSensor(SpecificBodyPositionSensor(['head']))

        #we changed sensors so we need to update environments sensorLength variable
        self.env.obsLen=len(self.env.getSensors())

        #normalization for the task spezifish sensors
        for i in range(self.env.obsLen-2*self.env.actLen):
            self.sensor_limits.append((-20, 20))
        self.epiLen=400 #suggested episode length for this task 
        self.maxHight=4.0 #maximum hight reached during episode
        self.maxPower=400.0 #jumping needs more power
        
    def getReward(self):
        # calculate reward and return reward
        reward=self.env.getSensorByName('SpecificBodyPositionSensor8')[1] #reward is hight of head
        if reward > self.maxHight: self.maxHight=reward
        if self.count==self.epiLen: reward=self.maxHight
        else: reward=0.0
        return reward

    def res(self):
        self.count = 0 
        self.reward_history.append(self.getTotalReward())
        self.maxHight=4.0

class StandingUpTask(StandingTask):
    def __init__(self, env):
        StandingTask.__init__(self, env)
        self.epiLen=2000 #suggested episode length for this task 
        self.env.tourqueList[0]=2.5
        self.env.tourqueList[1]=2.5
        
    def getReward(self):
        # calculate reward and return reward
        if self.count < 800:
            return 0.0
        else:
            reward=self.env.getSensorByName('SpecificBodyPositionSensor8')[1]/float(self.epiLen-800) #reward is hight of head
            if reward>4.0: reward=4.0 #to prevent jumping reward can't get bigger than head position while standing absolut upright
            return reward

    def performAction(self, action):
        if self.count < 800: 
            a=ones(self.env.actLen, int)*self.maxPower*self.env.tourqueList*0. #-1
            '''a[0]*=0. #-1.
            a[1]*=0. #-1.
            a[2]*=0.
            a[3]*=1.
            a[4]*=1.
            a[9]*=-1.
            a[10]*=-1.'''
            StandingTask.performAction(self, a)            
        else: StandingTask.performAction(self, action)        
