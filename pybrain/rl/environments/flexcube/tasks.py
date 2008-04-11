import random, math, os
from time import * 
from pybrain.rl.tasks import EpisodicTask
from scipy import ones
        
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
        self.maxSpeed=self.dif/100.0 
        self.picCount=0
    
    def incStep(self):
        self.step+=1
        self.epiStep+=1

    def getReward(self):
        # calculate reward and return self.reward
        self.reward[0]=self.rawReward
        return self.reward[:]

    def getObservation(self):
        # do something with self.sensors and return self.state
        aktSensors=self.env.getSensors()
        output=[]
        for i in aktSensors:
          for j in self.obsSensors:
            if i[0]==j:
              momSens=i[2:]
              for k in momSens:
                output.append(k/15.0-1.0)
          if i[0]==self.rewardSensor[0]:
            self.rawReward=i[2]
        output.append(math.sin(float(self.epiStep)/60.0))
        
        self.env.obsLen=len(output)    
        return output[:]       
    
    def performAction(self, action):
        """ a filtered mapping towards performAction of the underlying environment. """                
        # scaling
        self.incStep()
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
        self.obsSensors=["EdgesReal"]    
        self.inDim=len(self.getObservation())     
        self.plotString=["World Interactions", "Size", "Reward on Growing Task"]    
        
    def getReward(self):
        self.reward[0]=self.rawReward
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
        
    def getReward(self):
        if self.epiStep<2000: self.reward[0]=0.0
        else: self.reward[0]=self.rawReward
        return self.reward[0]

    def reset(self):
        self.reward[0]=0.0   
        self.rawReward=0.0         
        self.env.reset()
        self.action=[self.env.dists[0]]*self.outDim
        self.epiStep=0
        EpisodicTask.reset(self)

    def isFinished(self):
        return (self.epiStep>=2000)

class WalkToDirectionTask(NoRewardTask):
    def __init__(self, env):
        NoRewardTask.__init__(self, env)
        self.rewardSensor=["Target"]
        self.obsSensors=["EdgesTarget","EdgesReal","VerticesContact","Target"]    
        self.inDim=len(self.getObservation())     
        self.plotString=["World Interactions", "Distance", "Reward on Target Approach Task"]
        self.env.mySensors.sensors[5].targetList=[[-80.0,0.0,0.0]]
        self.epiLen=2000

    def getReward(self):
        self.reward[0]=(80.0-self.rawReward)/float(self.epiLen)
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

class TargetTask(NoRewardTask):
    def __init__(self, env):
        NoRewardTask.__init__(self, env)
        self.rewardSensor=["Target"]
        self.obsSensors=["EdgesTarget","EdgesReal","VerticesContact","Target"]    
        self.inDim=len(self.getObservation())     
        self.plotString=["World Interactions", "Distance", "Reward on Target Approach Task"]
        self.env.mySensors.sensors[5].targetList=[[-80.0,0.0,0.0]]
        if self.env.hasRenderer():
            self.env.getRenderer().target=[0.0,0.0,-80.0]

        self.epiLen=8000

    def getReward(self):
        self.reward[0]=(80.0-self.rawReward)/float(self.epiLen)
        return self.reward[0]

    def reset(self):
        self.reward[0]=0.0   
        self.rawReward=0.0         
        self.env.reset()
        self.action=[self.env.dists[0]]*self.outDim
        self.epiStep=0
        EpisodicTask.reset(self)

    def isFinished(self):
        if self.epiStep==self.epiLen/4:
            self.env.reset()
            self.env.mySensors.sensors[5].targetList=[[0.0,0.0,-80.0]]
            if self.env.hasRenderer(): self.env.getRenderer().target=[0.0,0.0,-80.0]
        if self.epiStep==self.epiLen/2:
            self.env.reset()
            self.env.mySensors.sensors[5].targetList=[[0.0,0.0,80.0]]
            if self.env.hasRenderer(): self.env.getRenderer().target=[0.0,0.0,80.0]
        if self.epiStep==3*self.epiLen/4:
            self.env.reset()
            self.env.mySensors.sensors[5].targetList=[[80.0,0.0,0.0]]
            if self.env.hasRenderer(): self.env.getRenderer().target=[80.0,0.0,0.0]
        return (self.epiStep>=self.epiLen)
