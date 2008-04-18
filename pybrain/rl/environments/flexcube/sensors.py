from scipy import sqrt, zeros, array, clip

class Sensors:
  def __init__(self, sensorList):
    self.sensors=[]
    for i in sensorList:
      self.sensors.append(eval(i+"()"))
    
  def updateSensor(self, pos, vel, dist, center, wEdges):
    for i in self.sensors:
      i.updateSensor(pos, vel, dist, center, wEdges)

  def getSensor(self):
    output=[]
    for i in self.sensors:
      output.append(i.getSensor()[:])
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
    self.targetList=[array([-80.0, 0.0, 0.0])]
    self.edges=array([1, 2, 4, 11, 13, 19, 22, 31, 37, 38, 47, 55])

  def updateSensor(self, pos, vel, dist, center, wEdges):
    self.pos=pos.copy()
    self.dist=dist.copy()
    self.centerOfGrav=center.copy().reshape(3)
    self.centerOfGrav[1]=0.0
    self.wantedEdges=wEdges.copy()

  def getSensor(self):
    return self.sensorOutput

class EdgesReal(defaultSensor):
  def getSensor(self):
    self.sensorOutput=["EdgesReal", 12]
    self.sensorOutput.append(self.dist[self.edges].reshape(12)/30.0)
    return self.sensorOutput

class EdgesSumReal(defaultSensor):
  def getSensor(self):
    self.sensorOutput=["EdgesSumReal", 1]
    self.sensorOutput.append((self.dist[self.edges].reshape(12)).sum(axis=0)-240.0)
    return self.sensorOutput

class EdgesTarget(defaultSensor):
  def getSensor(self):
    self.sensorOutput=["EdgesTarget", 12]
    self.sensorOutput.append(self.wantedEdges.reshape(12)/30.0) 
    return self.sensorOutput

class VerticesContact(defaultSensor):
  def getSensor(self):
    self.sensorOutput=["VerticesContact", 8]
    self.sensorOutput.append(clip((1.0-self.pos[:,1]), 0.0, 1.0))
    return self.sensorOutput

class VerticesMinHight(defaultSensor):
  def getSensor(self):
    self.sensorOutput=["VerticesMinHight", 1]
    self.sensorOutput.append(array([min(self.pos[:,1])]))
    return self.sensorOutput

class DistToOrigin(defaultSensor):
  def getSensor(self):
    self.sensorOutput=["DistToOrigin", 1]
    self.sensorOutput.append(array([sqrt((self.centerOfGrav**2).sum(axis=0))])) 
    return self.sensorOutput

class Target(defaultSensor):
  def getSensor(self):
    self.sensorOutput=["Target", 5]
    
    t=self.targetList[0]-self.centerOfGrav
    dist=sqrt((t**2).sum(axis=0))
    out=zeros(5, float)
    out[0]=dist

    for i in range(4):
        if i < 2:
            d=self.pos[i]-self.centerOfGrav
        else:
            d=self.pos[i+2]-self.centerOfGrav
        sen=sqrt((d**2).sum(axis=0))
        norm=dist*sen
        cosA = (d[0]*t[0]+d[2]*t[2])/norm
        sinA = (d[0]*t[2]-d[2]*t[0])/norm
        if cosA < 0.0: 
            if sinA > 0.0: sinA = 1.0
            else: sinA = -1.0
        out[i+1]=sinA
    self.sensorOutput.append(out)
    return self.sensorOutput
