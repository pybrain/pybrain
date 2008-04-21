import random, math, os
from time import * 
from pybrain.rl.tasks import EpisodicTask
from scipy import ones, array, c_, r_, sin, clip
import sensors
        
class NoRewardTask(EpisodicTask):
    ''' just a basic task, that doesn't return a reward '''
    def __init__(self, env):
        EpisodicTask.__init__(self, env)
        self.step=0
        self.epiStep=0
        self.action=[]
        self.reward=[0.0]
        self.rawReward=0.0
        self.obsSensors=["EdgesTarget","EdgesReal","VerticesContact"]
        self.rewardSensor=[""]
        self.oldDist=0.0
        self.plotString=["World Interactions", "Reward", "Reward on NoReward Task"]    
        self.inDim=len(self.getObservation())
        self.outDim=self.env.actLen        
        for i in range(self.outDim):
            self.action.append(self.env.dists[0])        
        self.dif=(self.env.fraktMax-self.env.fraktMin)*self.env.dists[0]
        self.maxSpeed=self.dif/30.0 
        self.picCount=0
    
    def incStep(self):
        self.step+=1
        self.epiStep+=1

    def getReward(self):
        # calculate reward and return self.reward
        self.reward[0]=self.rawReward-self.getPain()
        return self.reward[:]

    def getObservation(self):
        # do something with self.sensors and return self.state
        aktSensors=self.env.getSensors()
        output=array([])
        for i in aktSensors:
          for j in self.obsSensors:
            if i[0]==j:
              momSense=i[2]
              output=r_[output, momSense]
          if i[0]==self.rewardSensor[0]:
            self.rawReward=i[2][0]
          if i[0]=="EdgesReal":
            self.EdgeL=momSense.copy()
        return output[:]  

    def getPain(self):
        self.EdgeL=clip(self.EdgeL, 1.0, 4.0)
        return ((self.EdgeL-1.0)**2).sum(axis=0)          
    
    def performAction(self, action):
        """ a filtered mapping towards performAction of the underlying environment. """                
        # scaling
        self.incStep()
        #action=array([sin(float(self.epiStep)/5.0)]*12)
        
        action=(action+1.0)/2.0*self.dif+self.env.fraktMin*self.env.dists[0]
        actLen=len(action)
        for i in range(actLen):
            if action[i]<self.action[i]-self.maxSpeed:
                action[i]=self.action[i]-self.maxSpeed
            else:
                if action[i]>self.action[i]+self.maxSpeed:         
                    action[i]=self.action[i]+self.maxSpeed
        EpisodicTask.performAction(self, action)
        self.action=action[:] 
       
class GrowTask(NoRewardTask):
    def __init__(self, env):
        NoRewardTask.__init__(self, env)
        self.rewardSensor=["EdgesSumReal"]
        self.obsSensors=["EdgesReal", "EdgesTarget"]    
        self.inDim=len(self.getObservation())     
        self.plotString=["World Interactions", "Size", "Reward on Growing Task"]  
        self.env.mySensors=sensors.Sensors(self.obsSensors+self.rewardSensor)  
        
    def getReward(self):
        self.reward[0]=self.rawReward-self.getPain()
        return self.reward[0]

    def reset(self):
        self.reward[0]=0.0   
        self.rawReward=0.0         
        self.env.reset()
        self.action=[self.env.dists[0]]*self.outDim
        self.epiStep=0
        EpisodicTask.reset(self)

    def isFinished(self):
        return (self.epiStep>=200)

class WalkTask(NoRewardTask):
    def __init__(self, env):
        NoRewardTask.__init__(self, env)
        self.rewardSensor=["DistToOrigin"]
        self.obsSensors=["EdgesTarget","EdgesReal","VerticesContact"]    
        self.inDim=len(self.getObservation())     
        self.plotString=["World Interactions", "Distance", "Reward on Walking Task"]
        self.env.mySensors=sensors.Sensors(self.obsSensors+self.rewardSensor)  
        self.epiLen=2000
        
    def getReward(self):
        if self.epiStep<self.epiLen: self.reward[0]=-self.getPain()
        else: self.reward[0]=self.rawReward-self.getPain()
        return self.reward[0]

    def reset(self):
        self.reward[0]=0.0   
        self.rawReward=0.0         
        self.env.reset()
        self.action=[self.env.dists[0]]*self.outDim
        self.epiStep=0
        EpisodicTask.reset(self)

    def isFinished(self):
        return (self.epiStep>=self.epiLen)

class RollingUpTask(WalkTask):
    def __init__(self, env):
        WalkTask.__init__(self, env)
        self.env.startHight=200.0
        self.env.reset()

class WalkDirectionTask(WalkTask):
    def __init__(self, env):
        NoRewardTask.__init__(self, env)
        self.rewardSensor=["Target"]
        self.obsSensors=["EdgesTarget","EdgesReal","VerticesContact","Target"]    
        self.inDim=len(self.getObservation())     
        self.plotString=["World Interactions", "Distance", "Reward on Target Approach Task"]
        self.env.mySensors=sensors.Sensors(self.obsSensors)
        self.env.mySensors.sensors[3].targetList=[array([-80.0,0.0,0.0])]
        if self.env.hasRenderer(): self.env.getRenderer().target=self.env.mySensors.sensors[3].targetList[0]
        self.epiLen=2000

    def getReward(self):
        if self.epiStep<self.epiLen: self.reward[0]=-self.getPain()
        else: self.reward[0]=(80.0-self.rawReward)-self.getPain()
        return self.reward[0]

class TargetTask(WalkDirectionTask):
    def __init__(self, env):
        WalkDirectionTask.__init__(self, env)
        self.epiLen=6000

    def getReward(self):
        if self.epiStep==self.epiLen/3 or self.epiStep==2*self.epiLen/3 or self.epiStep==self.epiLen: 
            self.reward[0]=(80.0-self.rawReward)-self.getPain()
        else: self.reward[0]=-self.getPain()
        return self.reward[0]

    def isFinished(self):
        if self.epiStep==self.epiLen/3:
            self.env.reset()
            self.env.mySensors.sensors[3].targetList=[array([-56.6,0.0,-56.6])]
            if self.env.hasRenderer(): self.env.getRenderer().target=self.env.mySensors.sensors[3].targetList[0]
        if self.epiStep==2*self.epiLen/3:
            self.env.reset()
            self.env.mySensors.sensors[3].targetList=[array([-56.6,0.0,56.6])]
            if self.env.hasRenderer(): self.env.getRenderer().target=self.env.mySensors.sensors[3].targetList[0]
        return (self.epiStep>=self.epiLen)

class JumpTask(NoRewardTask):
    def __init__(self, env):
        NoRewardTask.__init__(self, env)
        self.rewardSensor=["VerticesMinHight"]
        self.obsSensors=["EdgesTarget","EdgesReal","VerticesContact"]    
        self.inDim=len(self.getObservation())     
        self.plotString=["World Interactions", "Distance", "Reward on Walking Task"]
        self.env.mySensors=sensors.Sensors(self.obsSensors+self.rewardSensor)  
        self.epiLen=500
        self.maxReward=0.0
        self.maxSpeed=self.dif/10.0

    def getReward(self):
        if self.epiStep<self.epiLen: 
            if self.rawReward > self.maxReward: self.maxReward = self.rawReward
            self.reward[0]=-self.getPain()
        else: self.reward[0]=self.maxReward-self.getPain()
        return self.reward[0]

    def reset(self):
        self.reward[0]=0.0   
        self.rawReward=0.0
        self.maxReward=0.0
        self.env.reset()
        self.action=[self.env.dists[0]]*self.outDim
        self.epiStep=0
        EpisodicTask.reset(self)

    def isFinished(self):
        return (self.epiStep>=self.epiLen)
