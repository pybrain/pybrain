__author__ = 'Martin Felder, felder@in.tum.de'

from math import sin, cos
import time
from scipy import random, eye, matrix, array

from pybrain.rl.environments.graphical import GraphicalEnvironment


class ShipSteeringEnvironment(GraphicalEnvironment):
    """ 
    Simulates an ocean going ship with substantial inertia in both forward
    motion and rotation, plus noise.
    
    State space (continuous):
        h       heading of ship in degrees (North=0)
        hdot    angular velocity of heading in degrees/minute
        v       velocity of ship in knots
    Action space (continuous):
        rudder  angle of rudder
        thrust  propulsion of ship forward
    """       
    
    # some (more or less) physical constants 
    dt = 4.        # simulated time (in seconds) per step
    mass = 1000.   # mass of ship in unclear units
    I = 1000.      # rotational inertia of ship in unclear units

    
    def __init__(self):
        GraphicalEnvironment.__init__(self)

        # initialize the environment (randomly)
        self.reset()
        self.action = [0.0, 0.0]
        self.delay = False
        
        
    def step(self):
        """ integrate state using simple rectangle rule """
        thrust, rudder = self.action
        h, hdot, v = self.sensors
        thrust = min(max(thrust,-1),+2)
        rudder = min(max(rudder,-90),+90)
        drag = 5*h + (rudder**2 + random.normal(0,1.0))
        force = 30.0*thrust - 2.0*v - 0.02*v*drag + random.normal(0,3.0)
        v = v + self.dt*force/self.mass
        v = min(max(v,-10),+40)
        torque = -v*(rudder + h + 1.0*hdot + random.normal(0,10.))
        last_hdot = hdot
        hdot += torque / self.I
        hdot = min(max(hdot,-180),180)
        h += (hdot + last_hdot) / 2.0
        if h>180.: 
            h -= 360.
        elif h<-180.: 
            h += 360.
        #print drag,force,torque
        self.sensors = (h,hdot,v)

        if self.hasRenderer():
            self.getRenderer().updateData(self.sensors)
            if self.delay: 
                time.sleep(self.tau)    
                        
    def reset(self):
        """ re-initializes the environment, setting the ship to rest at a random orientation.
        """
        #               [h,                           hdot, v]
        self.sensors = [random.uniform(-180., 180.), 0.0, 0.0]

    def getHeading(self):
        """ auxiliary access to just the heading, to be used by GoNorthwardTask """
        return self.sensors[0]
        
    def getSpeed(self):
        """ auxiliary access to just the speed, to be used by GoNorthwardTask """
        return self.sensors[2]
        
    
    def getSensors(self):
        """ returns the state one step (dt) ahead in the future. stores the state in
            self.sensors because it is needed for the next calculation. 
        """
        return self.sensors
                            
    def performAction(self, action):
        """ stores the desired action for the next time step.
        """
        self.action = action
        self.step()

    @property              
    def indim(self):
        return len(self.action)
    
    @property
    def outdim(self):
        return len(self.sensors)
    

