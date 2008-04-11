import math

class Sensors:
  def __init__(self, sensorList):
    self.sensors=[]
    for i in sensorList:
      self.sensors.append(eval(i+"()"))
    
  def updateSensor(self, points, wEdges):
    for i in self.sensors:
      i.updateSensor(points, wEdges)

  def getSensor(self):
    output=[]
    for i in self.sensors:
      output.append(i.getSensor()[:])
    return output

  def getSpezSensor(self, name):
    output=[]
    foundSensor=0
    for i in self.sensors:
      if i.getSensor()[0]==name:
        foundSensor=1
        output=i.getSensor()[:]
    if foundSensor==1:
      return output  
    else:
      print "Sensor not found: ", name      
      return output

  def getSensorList(self, sensList):
    output=[]
    for i in sensList:
      sens=self.getSpezSensor(i)[2:]
      for j in sens:
        output.append(j)
    return output             
  

class defaultSensor:
  def __init__(self):
    self.sensorOutput=["defaultSensor", 0]
    self.points=[]
    self.wantedEdges=[]
    self.actualEdges=[]
    self.targetList=[[-80.0, 0.0, 0.0]]

  def updateSensor(self, points, wEdges):
    self.points=points[:]
    self.wantedEdges=wEdges[:]

  def getSensor(self):
    return self.sensorOutput

  def dif(self, point1, point2):
    vect=[point1[0]-point2[0],point1[1]-point2[1],point1[2]-point2[2]]    
    dif=math.sqrt(vect[0]*vect[0]+vect[1]*vect[1]+vect[2]*vect[2])    
    return dif

class Smell(defaultSensor):
  def updateSensor(self, points, wEdges):
    self.points=points[:]
    self.wantedEdges=wEdges[:]

  def updateTarget(self, targetList=[[80.0,0.0,0.0]]):
    self.targetList=targetList[:]
  
  def getSensor(self):
    self.sensorOutput=["Smell", 3]
    kCount=0
    minDist=9999999999999.0
    maxK=0
    sensorP=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    for  k in self.points:
      for i in self.targetList:
        for j in range(3):
            sensorP[kCount]+=(i[j]-k.pos[j])*(i[j]-k.pos[j])        
      if sensorP[kCount]<minDist: 
        minDist=sensorP[kCount]
        maxK=kCount
      kCount+=1
    if maxK>=4: 
        self.sensorOutput.append(30.0)
        maxK-=4
    else: 
        self.sensorOutput.append(0.0)
    if maxK>=2: 
        self.sensorOutput.append(30.0)
        maxK-=2
    else: 
        self.sensorOutput.append(0.0)
    if maxK==1: 
        self.sensorOutput.append(30.0)
        maxK-=1
    else: 
        self.sensorOutput.append(0.0)
    if maxK!=0: print "Smell Sensor Error!!!!"
    return self.sensorOutput

class VerticesContact(defaultSensor):
  def getSensor(self):
    self.sensorOutput=["VerticesContact", 8]
    for i in range(8):
      if self.points[i].pos[1]<1.0:
        self.sensorOutput.append((1.0-self.points[i].pos[1])*30.0)
      else:
        self.sensorOutput.append(0.0)
    return self.sensorOutput

class VerticesMinHight(defaultSensor):
  def getSensor(self):
    self.sensorOutput=["VerticesMinHight", 1]
    minDist=1000.0
    for i in range(8):
      if self.points[i].pos[1]<minDist:
        minDist=self.points[i].pos[1]
    self.sensorOutput.append(minDist)
    return self.sensorOutput

class PosXAngle(defaultSensor):
  def getSensor(self):
    self.sensorOutput=["PosXAngle", 3]
    leftSum=[0.0,0.0,0.0]
    for i in range(4):
      for dim in range(3):
        leftSum[dim]+=self.points[i].pos[dim]
    rightSum=[0.0,0.0,0.0]
    for i in range(4):
      for dim in range(3):
        rightSum[dim]+=self.points[i+4].pos[dim]
    for dim in range(3):
        self.sensorOutput.append((rightSum-leftSum)/80.0)
    return self.sensorOutput

class EdgesTarget(defaultSensor):
  def getSensor(self):
    self.sensorOutput=["EdgesTarget", 12]
    for i in self.wantedEdges:
      self.sensorOutput.append(i) 
    if len(self.sensorOutput)!=self.sensorOutput[1]+2: print "Attention: SensorLength corrupt!", len(self.sensorOutput), self.sensorOutput[1]+2
    return self.sensorOutput

class EdgesReal(defaultSensor):
  def getSensor(self):
    self.sensorOutput=["EdgesReal", 12]
    self.prozessEdges()
    return self.sensorOutput

  def prozessEdges(self):
    for i in range(2):
      for j in range(2):
        for k in range(2):
          for i2 in range(2):
            for j2 in range(2):
              for k2 in range(2):
                sum=abs(i-i2)+abs(j-j2)+abs(k-k2)
                if sum==1 and i<=i2 and j<=j2 and k<=k2:
                  self.sensorOutput.append(self.dif(self.points[i*4+j*2+k].pos, self.points[i2*4+j2*2+k2].pos))

class DistToOrigin(defaultSensor):
  def getSensor(self):
    self.sensorOutput=["DistToOrigin", 1]
    
    centerOfGrav=[0.0,0.0,0.0]
    for i in range(2):
      for j in range(2):
        for k in range(2):
          for l in range(3):
            centerOfGrav[l]+=self.points[i*4+j*2+k].pos[l]
    sum=0.0
    for l in range(3):
      if l==1: l+=1
      centerOfGrav[l]/=8.0
      sum+=centerOfGrav[l]*centerOfGrav[l]
    sum=math.sqrt(sum)
    self.sensorOutput.append(sum) 
    return self.sensorOutput

class Target(defaultSensor):
  def getSensor(self):
    self.sensorOutput=["Target", 5]
    
    centerOfGrav=[0.0,0.0,0.0]
    for i in range(2):
      for j in range(2):
        for k in range(2):
          for l in range(3):
            centerOfGrav[l]+=self.points[i*4+j*2+k].pos[l]
    for l in range(3):
        centerOfGrav[l]/=8.0
    t=[0.0,0.0]
    t[0]=self.targetList[0][0]-centerOfGrav[0]
    t[1]=self.targetList[0][2]-centerOfGrav[2]
    dist=math.sqrt(t[0]*t[0]+t[1]*t[1])
    self.sensorOutput.append(dist) 

    d=[0.0,0.0]
    for i in range(4):
        if i < 2:
            d[0]=self.points[i].pos[0]-centerOfGrav[0]
            d[1]=self.points[i].pos[2]-centerOfGrav[2]
        else:
            d[0]=self.points[i+2].pos[0]-centerOfGrav[0]
            d[1]=self.points[i+2].pos[2]-centerOfGrav[2]
        sen=math.sqrt(d[0]*d[0]+d[1]*d[1])
        norm=dist*sen
        cosA = (d[0]*t[0]+d[1]*t[1])/norm
        sinA = (d[0]*t[1]-d[1]*t[0])/norm
        if cosA < 0.0: sinA /= abs(sinA)
        self.sensorOutput.append((sinA+1)*15.0)
    return self.sensorOutput

class EdgeSumTarget(defaultSensor):
  def getSensor(self):
    self.sensorOutput=["EdgesSumTarget", 1]
    
    sum=0.0
    for i in self.wantedEdges:
      sum+=i
    self.sensorOutput.append(sum) 
    return self.sensorOutput

class EdgeSumReal(defaultSensor):
  def getSensor(self):
    self.sensorOutput=["EdgesSumReal", 1]
    self.prozessEdges()
    return self.sensorOutput

  def prozessEdges(self):
    sum2=0.0
    for i in range(2):
      for j in range(2):
        for k in range(2):
          for i2 in range(2):
            for j2 in range(2):
              for k2 in range(2):
                sum=abs(i-i2)+abs(j-j2)+abs(k-k2)
                if sum==1 and i<=i2 and j<=j2 and k<=k2:
                  sum2+=self.dif(self.points[i*4+j*2+k].pos, self.points[i2*4+j2*2+k2].pos)                  
    self.sensorOutput.append(sum2-240.0)
