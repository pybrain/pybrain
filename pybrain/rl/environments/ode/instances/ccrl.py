from pybrain.rl.environments.serverInterface import GraphicalEnvironment
#from renderInterface import JohnnieRenderInterface
from pybrain.rl.environments.ode import *
from ode import collide
import imp
from scipy import array

class CCRLEnvironment(ODEEnvironment):
  def __init__(self, renderer=True, realtime=False, ip="127.0.0.1", port="21590", buf='16384'):
        ODEEnvironment.__init__(self, renderer, realtime, ip, port, buf)
        # load model file
        self.loadXODE(imp.find_module('pybrain')[1]+"/rl/environments/ode/models/ccrlTable.xode")

        # standard sensors and actuators    
        self.addSensor(sensors.JointSensor())
        self.addSensor(sensors.JointVelocitySensor()) 
        self.addActuator(actuators.JointActuator())
            
        #set act- and obsLength, the min/max angles and the relative max touques of the joints  
        self.actLen=self.getActionLength()
        self.obsLen=len(self.getSensors())
        #ArmLeft, ArmRight, Hip, PevelLeft, PevelRight, TibiaLeft, TibiaRight, KneeLeft, KneeRight, FootLeft, FootRight
        self.tourqueList=array([0.6, 0.6, 0.5, 0.5, 0.3, 0.3, 0.1, 0.1, 1.0, 1.0, 0.8, 0.8, 0.8, 0.5, 0.5, 0.1],)
        #self.tourqueList=array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],)
        self.cHighList=array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.9],)
        self.cLowList=array([-1.0, -1.0, -1.0, -1.5, -1.0, -1.0, -1.0, -0.7, -1.0, 0.0, -1.0, -1.5, -1.0, -1.0, -1.0, 0.0],)        

        self.stepsPerAction = 1
                
if __name__ == '__main__' :
    w = CCRLEnvironment() 
    while True:
        w.step()
        if w.stepCounter==1000: w.reset() 
