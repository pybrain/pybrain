__author__ = 'Frank Sehnke, sehnke@in.tum.de'

from time import sleep
from scipy import ones, array
from pybrain.utilities import threaded
from pybrain.tools.networking.udpconnection import UDPServer
import threading

class FlexCubeRenderInterface(object):
    def __init__(self, ip="127.0.0.1", port="21560"):
        self.target=array([80.0,0.0,0.0])
        #self.dataLock = threading.Lock()
        self.centerOfGrav=array([0.0,-2.0,0.0])
        self.points=ones((8,3),float)
        self.updateDone=True
        self.updateLock=threading.Lock()
        self.server=UDPServer(ip, port)
      
    @threaded()  
    def updateData(self, pos, cog):
        self.updateDone=False      
        if not self.updateLock.acquire(False): return
        self.points=pos.copy()
        self.centerOfGrav=cog.copy()
          
        # Listen for clients
        self.server.listen()
        if self.server.clients > 0: 
            # If there are clients send them the new data
            self.server.send([self.points, self.centerOfGrav, self.target])
        sleep(0.02)
        self.updateLock.release()
        self.updateDone=True
