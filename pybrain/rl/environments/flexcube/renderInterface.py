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
      self.cIP=[]
      self.addrList=[]
      self.UDPOutSockList=[]

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
        self.cIP.append(self.UDPInSock.recv(self.buf))
        self.addrList.append((self.cIP[0], self.outPort))
        self.UDPOutSockList.append(socket.socket(socket.AF_INET,socket.SOCK_DGRAM))
        self.clients+=1
      else:  
        self.UDPInSock.settimeout(2)
        try:
          cIP = self.UDPInSock.recv(self.buf)
          newClient=True
          for i in self.cIP:
              if cIP == i: 
                  newClient=False
                  break
          if newClient:
              self.cIP.append(cIP)
              self.addrList.append((self.cIP[self.clients], self.outPort))
              self.UDPOutSockList.append(socket.socket(socket.AF_INET,socket.SOCK_DGRAM))
              self.clients+=1    
        except:
          self.clients=0
          self.cIP=[]
          self.addrList=[]
          self.UDPOutSockList=[]
         
        sendString=""
        for i in self.points:
          for j in i:
              sendString+=repr(j)+" "
        for i in self.centerOfGrav:
          sendString+=repr(i)+" "
        count=0
        for i in self.UDPOutSockList:
            i.sendto(sendString, self.addrList[count])
            count+=1
      sleep(0.02)
      self.updateLock.release()
      self.updateDone=True
