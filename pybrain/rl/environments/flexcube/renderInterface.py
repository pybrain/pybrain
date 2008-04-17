#@PydevCodeAnalysisIgnore
from time import sleep
from scipy import ones, zeros, array

import socket
import string
import sys, os
from pybrain.utilities import threaded
import threading

class FlexCubeRenderInterface(object):
  def __init__(self):
      self.target=[80.0,0.0,0.0]
      #self.dataLock = threading.Lock()
      self.centerOfGrav=array([0.0,-2.0,0.0])
      self.points=ones((8,3),float)
      #TODO: read from config file
      host = '131.159.60.203'
      self.inPort = 21561
      self.outPort = 21560
      self.buf = 1024
      self.addr = (host,self.inPort)

      # Create socket and bind to address
      self.UDPInSock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM) #socket.SOCK_DGRAM
      self.UDPInSock.bind(self.addr)
      self.clients=0
      self.updateDone=True
      self.updateLock=threading.Lock()

  def setTarget(self, target):
    self.target=target[:]
  
  @threaded()  
  def updateData(self, pos, sensors):
      self.updateDone=False      
      if not self.updateLock.acquire(False): return
      self.points=pos.copy()
      self.centerOfGrav=sensors.copy()

      if self.clients<1:
        #listen for client
        self.UDPInSock.settimeout(None)        
        self.cIP = self.UDPInSock.recv(self.buf)
        self.addr = (self.cIP, self.outPort)
        self.UDPOutSock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM) #socket.SOCK_DGRAM
        self.clients+=1
      else:  
        self.UDPInSock.settimeout(2)
        try:
          self.cIP = self.UDPInSock.recv(self.buf)
        except:
          self.clients-=1
        sendString=""
        for i in self.points:
          for j in i:
              sendString+=repr(j)+" "
        for i in self.centerOfGrav:
          sendString+=repr(i)+" "
        self.UDPOutSock.sendto(sendString, self.addr)
      self.updateLock.release()
      self.updateDone=True
