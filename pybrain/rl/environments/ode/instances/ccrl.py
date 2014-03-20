__author__ = 'Frank Sehnke, sehnke@in.tum.de'

from pybrain.rl.environments.ode import ODEEnvironment, sensors, actuators
import imp
import xode #@UnresolvedImport
import ode #@UnresolvedImport
import sys
from scipy import array, asarray

class CCRLEnvironment(ODEEnvironment):
    def __init__(self, xodeFile="ccrlGlas.xode", renderer=True, realtime=False, ip="127.0.0.1", port="21590", buf='16384'):
        ODEEnvironment.__init__(self, renderer, realtime, ip, port, buf)
        # load model file
        self.pert = asarray([0.0, 0.0, 0.0])
        self.loadXODE(imp.find_module('pybrain')[1] + "/rl/environments/ode/models/" + xodeFile)

        # standard sensors and actuators
        self.addSensor(sensors.JointSensor())
        self.addSensor(sensors.JointVelocitySensor())
        self.addActuator(actuators.JointActuator())

        #set act- and obsLength, the min/max angles and the relative max touques of the joints
        self.actLen = self.indim
        self.obsLen = len(self.getSensors())
        #ArmLeft, ArmRight, Hip, PevelLeft, PevelRight, TibiaLeft, TibiaRight, KneeLeft, KneeRight, FootLeft, FootRight
        self.tourqueList = array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.5, 0.5, 0.1],)
        #self.tourqueList=array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],)
        self.cHighList = array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.9],)
        self.cLowList = array([-1.0, -1.0, -1.0, -1.5, -1.0, -1.0, -1.0, -0.7, -1.0, 0.0, -1.0, -1.5, -1.0, -1.0, -1.0, 0.0],)
        self.stepsPerAction = 1

    def step(self):
        # Detect collisions and create contact joints
        self.tableSum = 0
        self.glasSum = 0
        ODEEnvironment.step(self)

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
        tmpStr = geom2.name[:-2]
        handStr = geom1.name[:-1]
        if geom1.name == 'plate' and tmpStr != 'objectP':
            self.tableSum += len(contacts)
        if tmpStr == 'objectP' and handStr == 'pressLeft':
            if len(contacts) > 0: self.glasSum += 1
        tmpStr = geom1.name[:-2]
        handStr = geom2.name[:-1]
        if geom2.name == 'plate' and tmpStr != 'objectP':
            self.tableSum += len(contacts)
        if tmpStr == 'objectP' and handStr == 'pressLeft':
            if len(contacts) > 0: self.glasSum += 1

        # Create contact joints
        world, contactgroup = args
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
            j = ode.ContactJoint(world, contactgroup, c)
            j.name = None
            j.attach(geom1.getBody(), geom2.getBody())

    def loadXODE(self, filename, reload=False):
        """ loads an XODE file (xml format) and parses it. """
        f = file(filename)
        self._currentXODEfile = filename
        p = xode.parser.Parser()
        self.root = p.parseFile(f)
        f.close()
        try:
            # filter all xode "world" objects from root, take only the first one
            world = filter(lambda x: isinstance(x, xode.parser.World), self.root.getChildren())[0]
        except IndexError:
            # malicious format, no world tag found
            print("no <world> tag found in " + filename + ". quitting.")
            sys.exit()
        self.world = world.getODEObject()
        self._setWorldParameters()
        try:
            # filter all xode "space" objects from world, take only the first one
            space = filter(lambda x: isinstance(x, xode.parser.Space), world.getChildren())[0]
        except IndexError:
            # malicious format, no space tag found
            print("no <space> tag found in " + filename + ". quitting.")
            sys.exit()
        self.space = space.getODEObject()

        # load bodies and geoms for painting
        self.body_geom = []
        self._parseBodies(self.root)

        for (body, _) in self.body_geom:
            if hasattr(body, 'name'):
                tmpStr = body.name[:-2]
                if tmpStr == "objectP":
                    body.setPosition(body.getPosition() + self.pert)

        if self.verbosity > 0:
            print("-------[body/mass list]-----")
            for (body, _) in self.body_geom:
                try:
                    print(body.name, body.getMass())
                except AttributeError:
                    print("<Nobody>")

        # now parse the additional parameters at the end of the xode file
        self.loadConfig(filename, reload)

    def reset(self):
        ODEEnvironment.reset(self)
        self.pert = asarray([1.5, 0.0, 1.0])

if __name__ == '__main__' :
    w = CCRLEnvironment()
    while True:
        w.step()
        if w.stepCounter == 1000: w.reset()

