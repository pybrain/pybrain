__author__ = 'Martin Felder, felder@in.tum.de'

from scipy import random
from pybrain.tools.networking.udpconnection import UDPServer
import threading
from pybrain.utilities import threaded
from time import sleep

from pybrain.rl.environments.environment import Environment


class ShipSteeringEnvironment(Environment):
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

    def __init__(self, render=True, ip="127.0.0.1", port="21580", numdir=1):
        # initialize the environment (randomly)
        self.action = [0.0, 0.0]
        self.delay = False
        self.numdir = numdir  # number of directions in which ship starts
        self.render = render
        if self.render:
            self.updateDone = True
            self.updateLock = threading.Lock()
            self.server = UDPServer(ip, port)
        self.reset()

    def step(self):
        """ integrate state using simple rectangle rule """
        thrust = float(self.action[0])
        rudder = float(self.action[1])
        h, hdot, v = self.sensors
        rnd = random.normal(0, 1.0, size=3)

        thrust = min(max(thrust, -1), +2)
        rudder = min(max(rudder, -90), +90)
        drag = 5 * h + (rudder ** 2 + rnd[0])
        force = 30.0 * thrust - 2.0 * v - 0.02 * v * drag + rnd[1] * 3.0
        v = v + self.dt * force / self.mass
        v = min(max(v, -10), +40)
        torque = -v * (rudder + h + 1.0 * hdot + rnd[2] * 10.)
        last_hdot = hdot
        hdot += torque / self.I
        hdot = min(max(hdot, -180), 180)
        h += (hdot + last_hdot) / 2.0
        if h > 180.:
            h -= 360.
        elif h < -180.:
            h += 360.
        self.sensors = (h, hdot, v)

    def closeSocket(self):
        self.server.UDPInSock.close()
        sleep(10)

    def reset(self):
        """ re-initializes the environment, setting the ship to rest at a random orientation.
        """
        #               [h,                           hdot, v]
        self.sensors = [random.uniform(-30., 30.), 0.0, 0.0]
        if self.render:
            if self.server.clients > 0:
                # If there are clients send them reset signal
                self.server.send(["r", "r", "r"])

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
        if self.render:
            if self.updateDone:
                self.updateRenderer()
                if self.server.clients > 0:
                    sleep(0.2)

    @threaded()
    def updateRenderer(self):
        self.updateDone = False
        if not self.updateLock.acquire(False): return

        # Listen for clients
        self.server.listen()
        if self.server.clients > 0:
            # If there are clients send them the new data
            self.server.send(self.sensors)
        sleep(0.02)
        self.updateLock.release()
        self.updateDone = True

    @property
    def indim(self):
        return len(self.action)

    @property
    def outdim(self):
        return len(self.sensors)


