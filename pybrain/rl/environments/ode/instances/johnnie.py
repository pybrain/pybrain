from pybrain.rl.environments.ode import *
import imp
from scipy import array

class JohnnieEnvironment(ODEEnvironment):
  def __init__(self, render=True):
        ODEEnvironment.__init__(self, render)
        # load model file
        self.loadXODE(imp.find_module('pybrain')[1]+"/rl/environments/ode/models/johnnie.xode")             # load XML file that describes the world

        # standard sensors and actuators    
        self.addSensor(sensors.JointSensor())
        self.addSensor(sensors.JointVelocitySensor()) 
        self.addActuator(actuators.JointActuator())

        #Start renderer if set     
        if self.hasRenderer():
            self.getRenderer().setFrameRate(25) 
            self.getRenderer().start()        
        
        #set act- and obsLength, the min/max angles and the relative max touques of the joints  
        self.actLen=self.getActionLength()
        self.obsLen=len(self.getSensors())
        #ArmLeft, ArmRight, Hip, PevelLeft, PevelRight, TibiaLeft, TibiaRight, KneeLeft, KneeRight, FootLeft, FootRight
        self.tourqueList=array([0.2, 0.2, 0.2, 0.5, 0.5, 2.0, 2.0,2.0,2.0,0.5,0.5],)
        self.cHighList=array([1.0, 1.0, 0.5, 0.5, 0.5, 1.5, 1.5,1.5,1.5,0.25,0.25],)
        self.cLowList=array([-0.5, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0,0.0,0.0,-0.25,-0.25],)        

        self.stepsPerAction = 1
                
if __name__ == '__main__' :
    w = JohnnieEnvironment() 
    while True:
        w.step() 
