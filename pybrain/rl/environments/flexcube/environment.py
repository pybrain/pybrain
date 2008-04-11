#@PydevCodeAnalysisIgnore

import sys
import random, math
import sensors
from pybrain.rl.environments.graphical import GraphicalEnvironment
from renderer import FlexCubeRenderer
from massPoint import *

class FlexCubeEnvironment(GraphicalEnvironment):
  def __init__(self, renderer=True):
    # initialize base class
    GraphicalEnvironment.__init__(self)
    self.actLen=12
    self.obsLen=32
    self.mySensors=sensors.Sensors(["EdgesTarget", "EdgesReal","VerticesContact", "DistToOrigin", "EdgeSumReal", "Target"])  #,"EdgeSumReal", "VerticesMinHight", "Smell"
    self.points=[]
    for i in range(8):
      self.points.append(MassPoint())
    self.dists=[20.0, math.sqrt(2.0)*20, math.sqrt(3.0)*20]
    self.gravVect=[0.0,-0.04,0.0]
    self.centerOfGrav=[0.0,0.0,0.0]
    self.d=0.1
    self.startHight=11.0
    self.startX=0
    self.dumping=0.995
    self.xSpeed=0.0
    self.distMatrix=MArray()
    self.fraktMin=0.7
    self.fraktMax=1.3
    self.reset()
    self.euler()
    self.mySensors.updateSensor(self.points, self.action)
    if renderer:
        self.setRenderer(FlexCubeRenderer())
        self.getRenderer().updateData(self.points, [self.centerOfGrav])
    
  def reset(self):
    #TODO let this be very temporary!
    self.action=[]
    for i in range(self.actLen):
      self.action.append(self.dists[0])

    for i in range(2):
      for j in range(2):
        for k in range(2):
          self.points[i*4+j*2+k].pos=[i*self.dists[0]-self.dists[0]/2.0+self.startX,j*self.dists[0]-self.dists[0]/2.0+self.startHight,k*self.dists[0]-self.dists[0]/2.0]
          self.points[i*4+j*2+k].vel=[0.0,0.0,0.0]
          
    for i in range(2):
      for j in range(2):
        for k in range(2):
          for i2 in range(2):
            for j2 in range(2):
              for k2 in range(2):
                sum=abs(i-i2)+abs(j-j2)+abs(k-k2)
                index="%i,%i,%i,%i,%i,%i" % (i,j,k,i2,j2,k2)                  
                if sum>0:
                  self.distMatrix.set(index, self.dists[sum-1])
                else:
                  self.distMatrix.set(index, 0.0)
      
  def doNothing(self):
    self.newPic=0
    sleep(0.01)
    
  def setTarget(self, target):
    if self.hasRenderer(): 
      if self.getRenderer().isAlive():
        self.getRenderer().setTarget(target)    
    
  def performAction(self, action):
    action=self.normAct(action)[:]
    self.action=action[:]
    self.act(action, action, 1)
    self.euler()
    
    if self.hasRenderer(): 
      if self.getRenderer().isAlive():
        self.getRenderer().updateData(self.points, [self.centerOfGrav])
      
  def getSensors(self):
    self.mySensors.updateSensor(self.points, self.action)    
    sens=[]
    return self.mySensors.getSensor()[:]

  def normAct(self, s):
    for count in range(len(s)):
      if s[count]< self.dists[0]*self.fraktMin:
        s[count]= self.dists[0]*self.fraktMin
      if s[count]> self.dists[0]*self.fraktMax:
        s[count]= self.dists[0]*self.fraktMax
    return s  

  def getState(self):
    return self.distMatrix

  def dist(self, point1, point2):
    dif=math.sqrt(math.pow(point1[0]-point2[0],2)+math.pow(point1[1]-point2[1],2)+math.pow(point1[2]-point2[2],2))    
    return dif

  def difVect(self, point1, point2):
    vect=[point1[0]-point2[0],point1[1]-point2[1],point1[2]-point2[2]]    
    return vect

  def addVect(self, point1, point2):
    vect=[point1[0]+point2[0],point1[1]+point2[1],point1[2]+point2[2]]    
    return vect

  def vAbs(self, vect):
    dif=math.sqrt(vect[0]*vect[0]+vect[1]*vect[1]+vect[2]*vect[2])    
    return dif
    
  def velDif(self, vect,dif,soll):
    zug=self.d*(soll-dif)
    dif=[vect[0]/dif*zug,vect[1]/dif*zug,vect[2]/dif*zug]
    return dif

  def dumpVect(self, vect, fakt):
    for i in range(3):
      vect[i]*=fakt
    return vect
  
  def normVect(self, vect, norm):
    summe=0.0
    for i in range(3):
      summe+=vect[i]*vect[i]
    vect=self.dumpVect(vect, norm/math.sqrt(summe))
    return vect
  
  def clear(self):
    li=[0.0,0.0,0.0]
    return li

  def getVolume(self):
    sum=0.0
    for i in range(2):
      for j in range(2):
        for k in range(2):
          for n in range(3):
            i2=i
            j2=j
            k2=k
            if n==0:
              i2=1-i
            if n==1:
              j2=1-j
            if n==2:
              k2=1-k
               
            sum+=self.dist(self.points[i*4+j*2+k].pos, self.points[i2*4+j2*2+k2].pos)
            
    return sum          

  def getDist(self, con):
    for i in range(2):
      for j in range(2):
        for k in range(2):
          for i2 in range(2):
            for j2 in range(2):
              for k2 in range(2):
                sum=abs(i-i2)+abs(j-j2)+abs(k-k2)
                compare=[i,j,k,i2,j2,k2]
                if con==compare:
                  index="%i,%i,%i,%i,%i,%i" % (i,j,k,i2,j2,k2)
                  return self.distMatrix.get(index)
    
  def act(self, oldS, newS, step):
    count=0
    for i in range(2):
      for j in range(2):
        for k in range(2):
          for i2 in range(2):
            for j2 in range(2):
              for k2 in range(2):
                sum=abs(i-i2)+abs(j-j2)+abs(k-k2)
                if sum==1 and i<=i2 and j<=j2 and k<=k2:
                  index="%i,%i,%i,%i,%i,%i" % (i,j,k,i2,j2,k2)
                  self.distMatrix.set(index, oldS[count]+step*float(newS[count]-oldS[count]))
                  index="%i,%i,%i,%i,%i,%i" % (i2,j2,k2,i,j,k)
                  self.distMatrix.set(index, oldS[count]+step*float(newS[count]-oldS[count]))
                  count+=1

  def euler(self):
    self.calcPhysics=1
    #forces to velos
    #Gravity
    for i in range(2):
      for j in range(2):
        for k in range(2):
          self.points[i*4+j*2+k].vel=self.addVect(self.points[i*4+j*2+k].vel, self.gravVect)
    
    #Inner Forces
    for i in range(2):
      for j in range(2):
        for k in range(2):
          for i2 in range(2):
            for j2 in range(2):
              for k2 in range(2):
                sum=abs(i-i2)+abs(j-j2)+abs(k-k2)
                point1=self.points[i*4+j*2+k].pos
                point2=self.points[i2*4+j2*2+k2].pos

                if sum>0:
                  vect=self.difVect(point1,point2) 
                  dif=self.vAbs(vect)
                  index="%i,%i,%i,%i,%i,%i" % (i,j,k,i2,j2,k2)
                  self.points[i*4+j*2+k].vel=self.addVect(self.points[i*4+j*2+k].vel, self.velDif(vect,dif,self.distMatrix.get(index)))

    #velos to positions
    for i in range(2):
      for j in range(2):
        for k in range(2):
          self.points[i*4+j*2+k].pos=self.addVect(self.points[i*4+j*2+k].pos,self.points[i*4+j*2+k].vel)

    #Collisions and friction
    for i in range(2):
      for j in range(2):
        for k in range(2):            
          if self.points[i*4+j*2+k].pos[1]<0:
            self.points[i*4+j*2+k].pos[1]=0.0
            self.points[i*4+j*2+k].vel[1]=-self.points[i*4+j*2+k].vel[1]
            self.points[i*4+j*2+k].vel[0]=0.0
            self.points[i*4+j*2+k].vel[2]=0.0
          #else:
          #  if self.points[i*4+j*2+k].pos[1]<0.5:
          #    for l in range(3):
          #      self.points[i*4+j*2+k].vel[l]*=2.0*self.points[i*4+j*2+k].pos[1]

    #dumping
    self.centerOfGrav=self.clear()
    for i in range(2):
      for j in range(2):
        for k in range(2):
          self.points[i*4+j*2+k].vel=self.dumpVect(self.points[i*4+j*2+k].vel, self.dumping)
          self.centerOfGrav=self.addVect(self.centerOfGrav,self.points[i*4+j*2+k].pos)
    self.centerOfGrav=self.dumpVect(self.centerOfGrav, 1.0/8.0)
