__author__ = 'Martin Felder, felder@in.tum.de'

from math import sin, cos
import time
from scipy import random, eye, matrix, array
from scipy import weave

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

    
    def __init__(self, numdir=1):
        GraphicalEnvironment.__init__(self)

        # initialize the environment (randomly)
        self.reset()
        self.action = [0.0, 0.0]
        self.delay = False
        self.numdir = numdir  # number of directions in which ship starts
        
        
    def step(self):
        """ integrate state using simple rectangle rule """
        thrust = float(self.action[0])
        rudder = float(self.action[1])
        h, hdot, v = self.sensors
        ret = array([0.0,0.0,0.0])
        rnd = random.normal(0,1.0, size=3)
        
        # weave code: dt, I and mass have been hard-coded in line v=..!
        code = """  
            #line 45 "shipsteer.py"
            if (thrust < -1.0) { thrust = -1.0; }
            else if (thrust > 2.0) thrust = 2.0;
            if (rudder < -90.0) { rudder = -90.0; }
            else if (rudder > 90.0) rudder = 90.0;
            double drag = 5.*h + (rudder*rudder + rnd(0));
            double force = 30.0*thrust - 2.0*v - 0.02*v*drag + rnd(1)*3.0;
            v += force*4./1000.;
            if (v < -10.0) { v = -10.0; }
            else if (v > 40.0) v = 40.0;
            double torque = -v*(rudder + h + 1.0*hdot + rnd(2)*10.);
            double last_hdot = hdot;
            hdot += torque / 1000.;
            if (hdot < -180.0) { hdot = -180.0; }
            else if (hdot > 180.0) hdot = 180.0;
            h += (hdot + last_hdot) / 2.0;
            if (h < -180.0) { h += 360.0; }
            else if (h > 180.0) h -= 360.0;
            ret(0) = h;
            ret(1) = hdot;
            ret(2) = v;
            """
        # original code
        thrust = min(max(thrust,-1),+2)
        rudder = min(max(rudder,-90),+90)
        drag = 5*h + (rudder**2 + rnd[0])
        force = 30.0*thrust - 2.0*v - 0.02*v*drag + rnd[1]*3.0
        v = v + self.dt*force/self.mass
        v = min(max(v,-10),+40)
        torque = -v*(rudder + h + 1.0*hdot + rnd[2]*10.)
        last_hdot = hdot
        hdot += torque / self.I
        hdot = min(max(hdot,-180),180)
        h += (hdot + last_hdot) / 2.0
        if h>180.: 
            h -= 360.
        elif h<-180.: 
            h += 360.
        self.sensors = (h,hdot,v)

        #variables = 'thrust', 'rudder', 'h', 'hdot', 'v', 'ret', 'rnd'
        
        #weave.inline(
            #code, 
            #variables, 
            #type_converters=weave.converters.blitz, 
            #compiler='gcc')
        #self.sensors = ret
        
        if self.hasRenderer():
            self.getRenderer().updateData(self.sensors)
            if self.delay: 
                time.sleep(self.tau)    
                        
    def reset(self):
        """ re-initializes the environment, setting the ship to rest at a random orientation.
        """
        #               [h,                           hdot, v]
        self.sensors = [random.uniform(-30., 30.), 0.0, 0.0]

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
    

