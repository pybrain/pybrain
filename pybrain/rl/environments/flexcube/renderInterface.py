#@PydevCodeAnalysisIgnore
from time import sleep
from scipy import ones, array
from pybrain.utilities import threaded
from pybrain.tools.networking.udpconnection import UDPServer
import threading

class FlexCubeRenderInterface(object):
  def __init__(self, ip="127.0.0.1", port="21560"):
      self.target=[80.0,0.0,0.0]
      #self.dataLock = threading.Lock()
      self.centerOfGrav=array([0.0,-2.0,0.0])
      self.points=ones((8,3),float)
      self.updateDone=True
      self.updateLock=threading.Lock()
      self.server=UDPServer(ip, port)

  def setTarget(self, target):
    self.target=target[:]
  
  @threaded()  
  def updateData(self, pos, sensors):
      self.updateDone=False      
      if not self.updateLock.acquire(False): return
      self.points=pos.copy()
      self.centerOfGrav=sensors.copy()
      self.server.listen()
      if self.server.clients > 0: 
          self.server.send([self.points, self.centerOfGrav])
      sleep(0.02)
      self.updateLock.release()
      self.updateDone=True
