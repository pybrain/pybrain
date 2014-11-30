from __future__ import print_function

__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

import sys, time
from scipy import random, asarray
import xode.parser, xode.body, xode.geom #@UnresolvedImport @UnusedImport @Reimport
import ode #@UnresolvedImport

from pybrain.rl.environments.environment import Environment
from .tools.configgrab import ConfigGrabber
from . import sensors, actuators
from pybrain.utilities import threaded
import threading
import warnings
from pybrain.tools.networking.udpconnection import UDPServer

class ODEEnvironment(Environment):
    """
    Virtual simulation for rigid bodies. Uses ODE as a physics engine and OpenGL as graphics
    interface to simulate and display arbitrary objects. Virtual worlds are defined through
    an XML file in XODE format (see http://tanksoftware.com/xode/). This file can be loaded
    into the world with the "loadXODE(...)" function. Use performAction(...) or step() to advance
    the simulation by one step.
    """
    # objects with names within one tuple can pass each other
    passpairs = []
    # define a verbosity level for selective debug output (0=none)
    verbosity = 0

    def __init__(self, render=True, realtime=True, ip="127.0.0.1", port="21590", buf='16384'):
        """ initializes the virtual world, variables, the frame rate and the callback functions."""
        print("ODEEnvironment -- based on Open Dynamics Engine.")

        # initialize base class
        self.render = render
        if self.render:
            self.updateDone = True
            self.updateLock = threading.Lock()
            self.server = UDPServer(ip, port)
        self.realtime = realtime

        # initialize attributes
        self.resetAttributes()

        # initialize the textures dictionary
        self.textures = {}

        # initialize sensor and exclude list
        self.sensors = []
        self.excludesensors = []

        #initialize actuators list
        self.actuators = []

        # A joint group for the contact joints that are generated whenever two bodies collide
        self.contactgroup = ode.JointGroup()

        self.dt = 0.01
        self.FricMu = 8.0
        self.stepsPerAction = 1
        self.stepCounter = 0

    def closeSocket(self):
        self.server.UDPInSock.close()
        time.sleep(10)

    def resetAttributes(self):
        """resets the class attributes to their default values"""
        # initialize root node
        self.root = None

        # A list with (body, geom) tuples
        self.body_geom = []

    def reset(self):
        """resets the model and all its parameters to their original values"""
        self.loadXODE(self._currentXODEfile, reload=True)
        self.stepCounter = 0

    def setGravity(self, g):
        """set the world's gravity constant in negative y-direction"""
        self.world.setGravity((0, -g, 0))


    def _setWorldParameters(self):
        """ sets parameters for ODE world object: gravity, error correction (ERP, default=0.2),
        constraint force mixing (CFM, default=1e-5).  """
        self.world.setGravity((0, -9.81, 0))
        # self.world.setERP(0.2)
        # self.world.setCFM(1e-9)

    def _create_box(self, space, density, lx, ly, lz):
        """Create a box body and its corresponding geom."""
        # Create body and mass
        body = ode.Body(self.world)
        M = ode.Mass()
        M.setBox(density, lx, ly, lz)
        body.setMass(M)
        body.name = None
        # Create a box geom for collision detection
        geom = ode.GeomBox(space, lengths=(lx, ly, lz))
        geom.setBody(body)
        geom.name = None

        return (body, geom)

    def _create_sphere(self, space, density, radius):
        """Create a sphere body and its corresponding geom."""
        # Create body and mass
        body = ode.Body(self.world)
        M = ode.Mass()
        M.setSphere(density, radius)
        body.setMass(M)
        body.name = None

        # Create a sphere geom for collision detection
        geom = ode.GeomSphere(space, radius)
        geom.setBody(body)
        geom.name = None

        return (body, geom)

    def drop_object(self):
        """Drops a random object (box, sphere) into the scene."""
        # choose between boxes and spheres
        if random.uniform() > 0.5:
            (body, geom) = self._create_sphere(self.space, 10, 0.4)
        else:
            (body, geom) = self._create_box(self.space, 10, 0.5, 0.5, 0.5)
        # randomize position slightly
        body.setPosition((random.normal(-6.5, 0.5), 6.0, random.normal(-6.5, 0.5)))
        # body.setPosition( (0.0, 3.0, 0.0) )
        # randomize orientation slightly
        #theta = random.uniform(0,2*pi)
        #ct = cos (theta)
        #st = sin (theta)
        # rotate body and append to (body,geom) tuple list
        # body.setRotation([ct, 0., -st, 0., 1., 0., st, 0., ct])
        self.body_geom.append((body, geom))

    # -- sensor and actuator functions
    def addSensor(self, sensor):
        """ adds a sensor object to the list of sensors """
        if not isinstance(sensor, sensors.Sensor):
            raise TypeError("the given sensor is not an instance of class 'Sensor'.")
        # add sensor to sensors list
        self.sensors.append(sensor)
        # connect sensor and give it the virtual world object
        sensor._connect(self)

    def addActuator(self, actuator):
        """ adds a sensor object to the list of sensors """
        if not isinstance(actuator, actuators.Actuator):
            raise TypeError("the given actuator is not an instance of class 'Actuator'.")
        # add sensor to sensors list
        self.actuators.append(actuator)
        # connect actuator and give it the virtual world object
        actuator._connect(self)

    def addTexture(self, name, texture):
        """ adds a texture to the given ODE object name """
        self.textures[name] = texture

    def centerOn(self, name):
        return
        """ if set, keeps camera to the given ODE object name. """
        try:
            self.getRenderer().setCenterObj(self.root.namedChild(name).getODEObject())
        except KeyError:
            # name not found, unset centerObj
            print(("Warning: Cannot center on " + name))
            self.centerObj = None

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
            print(("no <world> tag found in " + filename + ". quitting."))
            sys.exit()
        self.world = world.getODEObject()
        self._setWorldParameters()
        try:
            # filter all xode "space" objects from world, take only the first one
            space = filter(lambda x: isinstance(x, xode.parser.Space), world.getChildren())[0]
        except IndexError:
            # malicious format, no space tag found
            print(("no <space> tag found in " + filename + ". quitting."))
            sys.exit()
        self.space = space.getODEObject()

        # load bodies and geoms for painting
        self.body_geom = []
        self._parseBodies(self.root)

        if self.verbosity > 0:
            print("-------[body/mass list]-----")
            for (body, _) in self.body_geom:
                try:
                    print((body.name, body.getMass()))
                except AttributeError:
                    print("<Nobody>")

        # now parse the additional parameters at the end of the xode file
        self.loadConfig(filename, reload)


    def loadConfig(self, filename, reload=False):
        # parameters are given in (our own brand of) config-file syntax
        self.config = ConfigGrabber(filename, sectionId="<!--odeenvironment parameters", delim=("<", ">"))

        # <passpairs>
        self.passpairs = []
        for passpairstring in self.config.getValue("passpairs")[:]:
            self.passpairs.append(eval(passpairstring))
        if self.verbosity > 0:
            print("-------[pass tuples]--------")
            print((self.passpairs))
            print("----------------------------")

        # <centerOn>
        # set focus of camera to the first object specified in the section, if any
        if self.render:
            try:
                self.centerOn(self.config.getValue("centerOn")[0])
            except IndexError:
                pass

        # <affixToEnvironment>
        for jointName in self.config.getValue("affixToEnvironment")[:]:
            try:
                # find first object with that name
                obj = self.root.namedChild(jointName).getODEObject()
            except IndexError:
                print(("ERROR: Could not affix object '" + jointName + "' to environment!"))
                sys.exit(1)
            if isinstance(obj, ode.Joint):
                # if it is a joint, use this joint to fix to environment
                obj.attach(obj.getBody(0), ode.environment)
            elif isinstance(obj, ode.Body):
                # if it is a body, create new joint and fix body to environment
                j = ode.FixedJoint(self.world)
                j.attach(obj, ode.environment)
                j.setFixed()

        # <colors>
        for coldefstring in self.config.getValue("colors")[:]:
            # ('name', (0.3,0.4,0.5))
            objname, coldef = eval(coldefstring)
            for (body, _) in self.body_geom:
                if hasattr(body, 'name'):
                    if objname == body.name:
                        body.color = coldef
                        break


        if not reload:
            # add the JointSensor as default
            self.sensors = []
            ## self.addSensor(self._jointSensor)

            # <sensors>
            # expects a list of strings, each of which is the executable command to create a sensor object
            # example: DistToPointSensor('legSensor', (0.0, 0.0, 5.0))
            sens = self.config.getValue("sensors")[:]
            for s in sens:
                try:
                    self.addSensor(eval('sensors.' + s))
                except AttributeError:
                    print((dir(sensors)))
                    warnings.warn("Sensor name with name " + s + " not found. skipped.")
        else:
            for s in self.sensors:
                s._connect(self)
            for a in self.actuators:
                a._connect(self)

    def _parseBodies(self, node):
        """ parses through the xode tree recursively and finds all bodies and geoms for drawing. """

        # body (with nested geom)
        if isinstance(node, xode.body.Body):
            body = node.getODEObject()
            body.name = node.getName()
            try:
                # filter all xode geom objects and take the first one
                xgeom = filter(lambda x: isinstance(x, xode.geom.Geom), node.getChildren())[0]
            except IndexError:
                return() # no geom object found, skip this node
            # get the real ode object
            geom = xgeom.getODEObject()
            # if geom doesn't have own name, use the name of its body
            geom.name = node.getName()
            self.body_geom.append((body, geom))

        # geom on its own without body
        elif isinstance(node, xode.geom.Geom):
            try:
                node.getFirstAncestor(ode.Body)
            except xode.node.AncestorNotFoundError:
                body = None
                geom = node.getODEObject()
                geom.name = node.getName()
                self.body_geom.append((body, geom))

        # special cases for joints: universal, fixed, amotor
        elif isinstance(node, xode.joint.Joint):
            joint = node.getODEObject()

            if type(joint) == ode.UniversalJoint:
                # insert an additional AMotor joint to read the angles from and to add torques
                # amotor = ode.AMotor(self.world)
                # amotor.attach(joint.getBody(0), joint.getBody(1))
                # amotor.setNumAxes(3)
                # amotor.setAxis(0, 0, joint.getAxis2())
                # amotor.setAxis(2, 0, joint.getAxis1())
                # amotor.setMode(ode.AMotorEuler)
                # xode_amotor = xode.joint.Joint(node.getName() + '[amotor]', node.getParent())
                # xode_amotor.setODEObject(amotor)
                # node.getParent().addChild(xode_amotor, None)
                pass
            if type(joint) == ode.AMotor:
                # do the euler angle calculations automatically (ref. ode manual)
                joint.setMode(ode.AMotorEuler)

            if type(joint) == ode.FixedJoint:
                # prevent fixed joints from bouncing to center of first body
                joint.setFixed()

        # recursive call for all child nodes
        for c in node.getChildren():
            self._parseBodies(c)

    def excludeSensors(self, exclist):
        self.excludesensors.extend(exclist)

    def getSensors(self):
        """ returns combined sensor data """
        output = []
        for s in self.sensors:
            if not s.name in self.excludesensors:
                output.extend(s.getValues())
        return asarray(output)

    def getSensorNames(self):
        return [s.name for s in self.sensors]

    def getActuatorNames(self):
        return [a.name for a in self.actuators]

    def getSensorByName(self, name):
        try:
            idx = self.getSensorNames().index(name)
        except ValueError:
            warnings.warn('sensor ' + name + ' is not in sensor list.')
            return []

        return self.sensors[idx].getValues()

    @property
    def indim(self):
        num = 0
        for a in self.actuators:
            num += a.getNumValues()
        return num

    def getActionLength(self):
        print("getActionLength() is deprecated. use property 'indim' instead.")
        return self.indim

    @property
    def outdim(self):
        return len(self.getSensors())

    def performAction(self, action):
        """ sets the values for all actuators combined. """
        pointer = 0
        for a in self.actuators:
            val = a.getNumValues()
            a._update(action[pointer:pointer + val])
            pointer += val

        for _ in range(self.stepsPerAction):
            self.step()

    def getXODERoot(self):
        return self.root


    #--- callback functions ---#
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

    def getCurrentStep(self):
        return self.stepCounter

    @threaded()
    def updateClients(self):
        self.updateDone = False
        if not self.updateLock.acquire(False):
            return

        # build message to send
        message = []
        for (body, geom) in self.body_geom:
            item = {}
            # real bodies (boxes, spheres, ...)
            if body != None:
                # transform (rotate, translate) body accordingly
                item['position'] = body.getPosition()
                item['rotation'] = body.getRotation()
                if hasattr(body, 'color'): item['color'] = body.color

                # switch different geom objects
                if type(geom) == ode.GeomBox:
                    # cube
                    item['type'] = 'GeomBox'
                    item['scale'] = geom.getLengths()
                elif type(geom) == ode.GeomSphere:
                    # sphere
                    item['type'] = 'GeomSphere'
                    item['radius'] = geom.getRadius()

                elif type(geom) == ode.GeomCCylinder:
                    # capped cylinder
                    item['type'] = 'GeomCCylinder'
                    item['radius'] = geom.getParams()[0]
                    item['length'] = geom.getParams()[1] - 2 * item['radius']

                elif type(geom) == ode.GeomCylinder:
                    # solid cylinder
                    item['type'] = 'GeomCylinder'
                    item['radius'] = geom.getParams()[0]
                    item['length'] = geom.getParams()[1]
                else:
                    # TODO: add other geoms here
                    pass

            else:
                # no body found, then it must be a plane (we only draw planes)
                if type(geom) == ode.GeomPlane:
                    # plane
                    item['type'] = 'GeomPlane'
                    item['normal'] = geom.getParams()[0] # the normal vector to the plane
                    item['distance'] = geom.getParams()[1] # the distance to the origin

            message.append(item)

        # Listen for clients
        self.server.listen()
        if self.server.clients > 0:
            # If there are clients send them the new data
            self.server.send(message)
        time.sleep(0.02)
        self.updateLock.release()
        self.updateDone = True

    def step(self):
        """ Here the ode physics is calculated by one step. """

        # call additional callback functions for all kinds of tasks (e.g. printing)
        self._printfunc()

        # Detect collisions and create contact joints
        self.space.collide((self.world, self.contactgroup), self._near_callback)

        # Simulation step
        self.world.step(float(self.dt))
        # Remove all contact joints
        self.contactgroup.empty()

        # update all sensors
        for s in self.sensors:
            s._update()

        # update clients
        if self.render and self.updateDone:
            self.updateClients()
            if self.server.clients > 0 and self.realtime:
                time.sleep(self.dt)

        # increase step counter
        self.stepCounter += 1
        return self.stepCounter

    def _printfunc (self):
        pass
        # print(self.root.namedChild('palm').getODEObject().getPosition())

    def specialkeyfunc(self, c, x, y):
        """Derived classes can implement extra functionality here"""
        pass


    #--- helper functions ---#
    def _print_help(self):
        """ prints out the keyboard shortcuts. """
        print("v   -> toggle view with mouse on/off")
        print("s   -> toggle screen capture on/off")
        print("d   -> drop an object")
        print("f   -> lift all objects")
        print("m   -> toggle mouse view (press button to zoom)")
        print("r   -> random torque at all joints")
        print("a/z -> negative/positive torque to all joints")
        print("g   -> print current state")
        print("n   -> reset environment")
        self.specialfunctionDoc()
        print("x,q -> exit program")

    def specialfunctionDoc(self):
        """Derived classes can implement extra functionality here"""
        pass


#--- main function, if called directly ---

if __name__ == '__main__' :
    """
    little example on how to use the virtual world.
    Synopsis: python environment.py [modelname]
    Parameters: modelname = base name of the xode file to use (default: johnnie)
    """

    print("ODEEnvironment -- test program")
    if len(sys.argv) > 1:
        modelName = sys.argv[1]
    else:
        modelName = "johnnie"

    # initialize world and renderer and attach renderer to world
    w = ODEEnvironment()
    # load model file
    w.loadXODE("models/" + modelName + ".xode")             # load XML file that describes the world

    w.addSensor(sensors.JointSensor())
    w.addActuator(actuators.JointActuator())

    # start simulating the world
    while True:
        w.step()

