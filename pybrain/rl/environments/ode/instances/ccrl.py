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
        self.tourqueList=array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.5, 0.5, 0.1],)
        #self.tourqueList=array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],)
        self.cHighList=array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.9],)
        self.cLowList=array([-1.0, -1.0, -1.0, -1.5, -1.0, -1.0, -1.0, -0.7, -1.0, 0.0, -1.0, -1.5, -1.0, -1.0, -1.0, 0.0],)
        self.stepsPerAction = 1

  def step(self):
        """ Here the ode physics is calculated by one step. """

        # call additional callback functions for all kinds of tasks (e.g. printing)
        self._printfunc()

        # Detect collisions and create contact joints
        self.tableSum=0
        self.glasSum=0
        self.space.collide((self.world, self.contactgroup), self._near_callback)
        #print self.tableSum,         

        # Simulation step
        self.world.step(float(self.dt))
        # Remove all contact joints
        self.contactgroup.empty()
            
        # update all sensors
        for s in self.sensors:
            s._update()
        
        # increase step counter
        self.stepCounter += 1
        return self.stepCounter

  def _near_callback(self, args, geom1, geom2):
        """Callback function for the collide() method.
        This function checks if the given geoms do collide and
        creates contact joints if they do."""

        # only check parse list, if objects have name
        if geom1.name != None and geom2.name != None:
            # Preliminary checking, only collide with certain objects
            for p in self.passpairs:
                g1 = False
                g2 = False
                for x in p:
                    g1 = g1 or (geom1.name.find(x) != -1)
                    g2 = g2 or (geom2.name.find(x) != -1)
                if g1 and g2:
                    return()

        # Check if the objects do collide
        contacts = ode.collide(geom1, geom2)
        if geom1.name == 'plate' and geom2.name != 'glas': 
            self.tableSum+=len(contacts)
        if geom2.name == 'plate' and geom1.name != 'glas': 
            self.tableSum+=len(contacts)
        if geom1.name == 'glas' and (geom2.name == 'palmLeft' or geom2.name == 'fingerLeft1'  or geom2.name == 'fingerLeft2'): 
            if len(contacts) > 0: self.glasSum+=1
        if geom2.name == 'glas' and (geom1.name == 'palmLeft' or geom1.name == 'fingerLeft1'  or geom1.name == 'fingerLeft2'): 
            if len(contacts) > 0: self.glasSum+=1
        
        # Create contact joints
        world,contactgroup = args
        for c in contacts:
            p = c.getContactGeomParams()
            # parameters from Niko Wolf
            c.setBounce(0.2)
            c.setBounceVel(0.05) #Set the minimum incoming velocity necessary for bounce
            c.setSoftERP(0.6) #Set the contact normal "softness" parameter
            c.setSoftCFM(0.00005) #Set the contact normal "softness" parameter
            c.setSlip1(0.02) #Set the coefficient of force-dependent-slip (FDS) for friction direction 1
            c.setSlip2(0.02) #Set the coefficient of force-dependent-slip (FDS) for friction direction 2
            c.setMu(self.FricMu) #Set the Coulomb friction coefficient
            j = ode.ContactJoint(self.world, self.contactgroup, c)
            j.name = None
            j.attach(geom1.getBody(), geom2.getBody())
                
if __name__ == '__main__' :
    w = CCRLEnvironment() 
    while True:
        w.step()
        if w.stepCounter==1000: w.reset() 
