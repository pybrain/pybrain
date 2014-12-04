from __future__ import print_function

__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

import sys
from .xmltools import XMLstruct
from math import asin, cos, sin, pi, degrees, radians, pow
from scipy import array, matrix, sqrt
import random

class XODEfile(XMLstruct):
    """
    Creates a (virtual) XODE file, into which bodies, joints and custom
    parameters can be inserted. This file can be merged at a defined level
    with other instances of itself, and written to disk in the
    standard format.

    $Id:xodetools.py 150 2007-04-11 13:42:47Z ruecksti $
    """

    def __init__(self, name, **kwargs):
        """initialize the XODE structure with a name and the world and
        space tags"""
        self._xodename = name
        self._centerOn = None
        self._affixToEnvironment = None
        # sensors is a list of ['type', [args], {kwargs}]
        self.sensors = []
        # sensor elements is a list of joints to be used as pressure sensors
        self.sensorElements = []
        self._nSensorElements = 0
        self._pass = {}     # dict of sets containing objects allowed to pass
        self._colors = []   # list of tuples ('name', (r,g,b))
        XMLstruct.__init__(self, 'world')
        self.insert('space')
        # TODO: insert palm, support, etc. (derived class)

    def _mass2dens(self, shape, size, mass):
        """converts a mass into a density"""
        if shape == 'box':
            return mass / float(size[0] * size[1] * size[2])
        elif shape == 'cylinder' or shape == 'cappedCylinder':
            if shape == 'cylinder':
                return mass / (12.56637061 * size[0] * size[0] * size[1])
            else:
                return mass / (size[0] * size[0] * (12.56637061 * (size[1] - 2 * size[0]) + 4.18879020 * size[0]))
        elif shape == 'sphere':
            return mass / (4.18879020 * pow(size[0], 3))
        else:
            print(("Unknown shape: " + shape + " not implemented!"))
            sys.exit(1)

    def _dens2mass(self, shape, size, dens):
        """converts a density into a mass"""
        if shape == 'box':
            return dens * float(size[0] * size[1] * size[2])
        elif shape == 'cylinder' or shape == 'cappedCylinder':
            if shape == 'cylinder':
                return dens * (12.56637061 * size[0] * size[0] * size[1])
            else:
                return dens * (size[0] * size[0] * (12.56637061 * (size[1] - 2 * size[0]) + 4.18879020 * size[0]))
        elif shape == 'sphere':
            return dens * (4.18879020 * pow(size[0], 3))
        else:
            print(("Unknown shape: " + shape + " not implemented!"))
            sys.exit(1)

            
    def insertBody(self, bname, shape, size, density, pos=[0, 0, 0], passSet=None, euler=None, mass=None, color=None):
        """Inserts a body with the given custom name and one of the standard
        shapes. The size and pos parameters are given as xyz-lists or tuples.
        euler are three rotation angles (degrees), 
        if mass is given, density is calculated automatically"""
        self.insert('body', {'name': bname})
        if color is not None:
            self._colors.append((bname, color))
        self.insert('transform')
        self.insert('position', {'x':pos[0], 'y':pos[1], 'z':pos[2]})
        if euler is not None:
            self.up()
            self.insert('rotation')
            self.insert('euler', {'x':euler[0], 'y':euler[1], 'z':euler[2], 'aformat':'degrees'})
            self.up()
        self.up(2)
        self.insert('mass')
        if shape == 'box':
            dims = {'sizex':size[0], 'sizey':size[1], 'sizez':size[2]}
        elif shape == 'cylinder' or shape == 'cappedCylinder':
            dims = {'radius':size[0], 'length':size[1]}
        elif shape == 'sphere':
            dims = {'radius':size[0]}
        else:
            print(("Unknown shape: " + shape + " not implemented!"))
            sys.exit(1)
        if mass is not None:
            density = self._mass2dens(shape, size, mass)
            
        self.insert('mass_shape', {'density': density})
        self.insert(shape, dims)
        self.up(3)
        self.insert('geom')
        self.insert(shape, dims)
        self.up(3)
        # add the body to a matching pass set
        if passSet is not None:
            for pset in passSet:
                try:
                    self._pass[pset].add(bname)
                except KeyError:
                    self._pass[pset] = set([bname])



    def insertJoint(self, body1, body2, type, axis=None, anchor=(0, 0, 0), rel=False, name=None):
        """Inserts a joint of given type linking the two bodies. Default name is
        a "_"-concatenation of the body names. The anchor is a xyz-tuple, rel is 
        a boolean specifying whether the anchor coordinates refer to the body's origin, 
        axis parameters have to be provided as a dictionary."""
        if name is None: name = body1 + "_" + body2
        if rel:
            abs = 'false'
        else:
            abs = 'true'
            
        self.insert('joint', {'name': name })
        self.insert('link1', {'body': body1})
        self.up()
        self.insert('link2', {'body': body2})
        self.up()
        self.insert(type)
        if type == 'fixed':
            self.insert(None)  # empty subtag, seems to be needed by xode parser
        elif type == 'ball':
            self.insert('anchor', {'x':anchor[0], 'y':anchor[1], 'z':anchor[2], 'absolute':abs})
            self.up()
        elif type == 'slider':
            self.insert('axis', axis)
            self.up()
        elif type == 'hinge':
            self.insert('axis', axis)
            self.up()
            self.insert('anchor', {'x':anchor[0], 'y':anchor[1], 'z':anchor[2], 'absolute':abs})
            self.up()
        else:
            print(("Sorry, joint type " + type + " not yet implemented!"))
            sys.exit()
        self.up(2)
        return name


    def insertFloor(self, y= -0.5):
        """inserts a bodiless floor at given y offset"""
        self.insert('geom', {'name': 'floor'})
        self.insert('plane', {'a': 0, 'b': 1, 'c': 0, 'd': y})
        self.up(2)

    def insertPressureSensorElement(self, parent, name=None, shape='cappedCylinder', size=[0.16, 0.5], pos=[0, 0, 0], euler=[0, 0, 0], dens=1, \
                             mass=None, passSet=[], stiff=10.0):
        """Insert one single pressure sensor element of the given shape, size, density, etc.
        The sliding axis is by default oriented along the z-axis, which is also the default for cylinder shapes.
        You have to rotate the sensor into the correct orientation - the sliding axis will be rotated accordingly.
        Stiffness of the sensor's spring is set via stiff, whereby damping is calculated automatically to prevent
        oscillations."""
        
        if name is None: 
            name = 'psens' + str(self._nSensorElements)
        if mass is None:
            mass = self._dens2mass(shape, size, dens)
        else:
            dens = self._mass2dens(shape, size, mass)
            
        self._nSensorElements += 1        
        h = 0.02  # temporal stepwidth

        self.insertBody(name, shape, size, dens, pos=pos, euler=euler, passSet=passSet)

        # In the aperiodic limit case, we have 
        # kp = a kd^2 / 4m     i.e.    kd = 2 sqrt(m kp/a)
        # where kp is the spring constant, kd is the dampening constant, and m is the mass of the oscillator.
        # For practical purposes, it is often better if kp is a few percent stronger such that the sensor
        # resets itself faster (but still does not oscillate a lot), thus a~=1.02.
        # ERP = h kp / (h kp + kd)
        # CFM = 1 / (h kp + kd) = ERP / h kp
        # For assumed mass of finger of 0.5kg, kp=10, kd=4.5 is approx. the non-oscillatory case.
        kd = 2.0 * sqrt(mass * stiff / 1.02) 
        ERP = h * stiff / (h * stiff + kd)
        CFM = ERP / (h * stiff)
        
        # Furthermore, compute the sliding axis direction from the Euler angles (x-convention, see
        # http://mathworld.wolfram.com/EulerAngles.html): Without rotation, the axis is along
        # the z axis, just like a cylinder's axis
        w = array(euler) * pi / 180
        #A  = matrix([[cos(w[0]), sin(w[0]), 0], [-sin(w[0]),cos(w[0]),0],[0,0,1]])
        #A = matrix([[1,0,0], [0,cos(w[1]), sin(w[1])], [0, -sin(w[1]),cos(w[1])]]) * A
        #A = matrix([[cos(w[2]), sin(w[2]), 0], [-sin(w[2]),cos(w[2]),0],[0,0,1]]) * A
        # hmmm, it seems XODE is rather using the y-convention here:
        A = matrix([[-sin(w[0]), cos(w[0]), 0], [-cos(w[0]), -sin(w[0]), 0], [0, 0, 1]])
        A = matrix([[1, 0, 0], [0, cos(w[1]), sin(w[1])], [0, -sin(w[1]), cos(w[1])]]) * A
        A = matrix([[sin(w[2]), -cos(w[2]), 0], [cos(w[2]), sin(w[2]), 0], [0, 0, 1]]) * A
        ax = ((A * matrix([0, 0, 1]).getT()).flatten().tolist())[0]
        
        jname = self.insertJoint(name, parent, 'slider', \
            axis={'x':ax[0], 'y':ax[1], 'z':ax[2], "HiStop":0.0, "LowStop":0.0, "StopERP":ERP, "StopCFM":CFM })
        self.sensorElements.append(jname)
        return name
        
    def attachSensor(self, type, *args, **kwargs):
        """adds a sensor with the given type, arguments and keywords, see sensors module for details"""
        self.sensors.append([type, args, kwargs])
        
        
    def merge(self, xodefile, joinLevel='space'):
        """Merge a second XODE file into this one, at the specified
        level (which must exist in both files). The passpair lists are
        also joined. Upon return, the current tag for both objects
        is the one given."""
        self.top()
        if not self.downTo(joinLevel):
            print(("Error: Cannot merge " + self.name + " at level " + joinLevel))
        xodefile.top()
        if not xodefile.downTo(joinLevel):
            print(("Error: Cannot merge " + xodefile.name + " at level " + joinLevel))
        self.insertMulti(xodefile.getCurrentSubtags())
        self._pass.update(xodefile.getPassList())

    def centerOn(self, name):
        self._centerOn = name


    def affixToEnvironment(self, name):
        self._affixToEnvironment = name

    def getPassList(self):
        return(self._pass)

    def scaleModel(self, sc):
        """scales all spatial dimensions by the given factor
        FIXME: quaternions may cause problems, which are currently ignored"""
        # scale these attributes...
        scaleset = set(['x', 'y', 'z', 'a', 'b', 'c', 'd', 'sizex', 'sizey', 'sizez', 'length', 'radius']) 
        # ... unless contained in these tags (in which case they specify angles)
        exclude = set(['euler', 'finiteRotation', 'axisangle'])
        self.scale(sc, scaleset, exclude)
        
        
    def writeCustomParameters(self, f):
        """writes our custom parameters into an XML comment"""
        f.write('<!--odeenvironment parameters\n')
        if len(self._pass) > 0:
            f.write('<passpairs>\n')
            for pset in self._pass.values():
                f.write(str(tuple(pset)) + '\n')
        if self._centerOn is not None:
            f.write('<centerOn>\n')
            f.write(self._centerOn + '\n')
        if self._affixToEnvironment is not None:
            f.write('<affixToEnvironment>\n')
            f.write(self._affixToEnvironment + '\n')
        if self._nSensorElements > 0:
            f.write('<sensors>\n')
            f.write("SpecificJointSensor(" + str(self.sensorElements) + ",name='PressureElements')\n")
        # compile all sensor commands
        for sensor in self.sensors:
            outstr = sensor[0] + "("
            for val in sensor[1]:
                outstr += ',' + repr(val)
            for key, val in sensor[2].items():
                outstr += ',' + key + '=' + repr(val)
            outstr = outstr.replace('(,', '(') + ")\n"
            f.write(outstr)
        if self._colors:
            f.write('<colors>\n')
            for col in self._colors:
                f.write(str(col) + "\n")

        f.write('<end>\n')
        f.write('-->\n')


    def writeXODE(self, filename=None):
        """writes the created structure (plus header and footer) to file with
        the given basename (.xode is appended)"""
        if filename is None: filename = self._xodename
        f = file(filename + '.xode', 'wb')  # <-- wb here ensures Linux compatibility
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<xode version="1.0r23" name="' + self._xodename + '"\n')
        f.write('xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://tanksoftware.com/xode/1.0r23/xode.xsd">\n\n')
        self.write(f)
        f.write('</xode>\n')
        self.writeCustomParameters(f)
        f.close()
        print(("Wrote " + filename + '.xode'))

class XODEfinger(XODEfile):

    def __init__(self, name, **kwargs):
        """Creates one finger on a fixed palm, and adds some sensors"""
        XODEfile.__init__(self, name, **kwargs)
        # create the hand and finger
        self.insertBody('palm', 'box', [10, 2, 10], 5, pos=[3.75, 4, 0], passSet=['pal'])
        self.insertBody('sample', 'box', [10, 0.2, 40], 5, pos=[0, 0.4, 0], passSet=['sam'])
        self.insertJoint('palm', 'sample', 'fixed', name='palm_support')
        self.insertJoint('palm', 'sample', 'slider', axis={'x':0, 'y':0, 'z':1, 'FMax':11000.0, "HiStop":100.0, "LowStop":-100.0})
        self.insertBody('finger1_link0', 'cappedCylinder', [1, 7.5], 5, pos=[0, 4, 8.75], passSet=['pal', 'f1'])
        self.insertBody('finger1_link1', 'cappedCylinder', [1, 4], 5, pos=[0, 4, 14.5], passSet=['f1', 'f2'])
        self.insertBody('fingertip', 'cappedCylinder', [1, 2.9], 5, pos=[0, 4, 17.95], passSet=['f2', 'haptic'])
        self.insertJoint('palm', 'finger1_link0', 'hinge', \
            axis={'x':-1, 'y':0, 'z':0, "HiStop":8, "LowStop":0.0}, anchor=(0, 4, 5))
        self.insertJoint('finger1_link0', 'finger1_link1', 'hinge', \
            axis={'x':-1, 'y':0, 'z':0, "HiStop":15, "LowStop":0.0}, anchor=(0, 4, 12.5))
        self.insertJoint('finger1_link1', 'fingertip', 'hinge', \
            axis={'x':-1, 'y':0, 'z':0, "HiStop":15, "LowStop":0.0}, anchor=(0, 4, 16.5))
        self.centerOn('fingertip')
        self.affixToEnvironment('palm_support')
        self.insertFloor()
        # add one group of haptic sensors
        self._nSensorElements = 0
        self.sensorElements = []
        self.sensorGroupName = None
        self.insertHapticSensors()
        # give some structure to the sample
        self.insertSampleStructure(**kwargs)

    def insertHapticSensorsRandom(self):
        """insert haptic sensors at random locations"""
        self.sensorGroupName = 'haptic'
        for _ in range(5):
            self.insertHapticSensor(dx=random.uniform(-0.65, 0.65), dz=random.uniform(-0.4, 0.2))
        ##self.insertHapticSensor(dx=-0.055)


    def insertHapticSensors(self):
        """insert haptic sensors at predetermined locations
        (check using testhapticsensorslocations.py)"""
        self.sensorGroupName = 'haptic'
        x = [0.28484253596392306, -0.59653176701550947, -0.36877718203650889, 0.50549219349016294, -0.22467390532644882, 0.051978612692656596, -0.18287341960589126, 0.40477910340060383, 0.56041266484490182, -0.47806390012776134]
        z = [-0.20354546253333045, -0.23178541627964597, 0.04632154813480549, -0.27525024891443889, -0.20352571063065863, -0.07930554411063101, 0.025260779785407084, 0.091906227805625964, -0.031751424859005839, -0.0034220681106161277]
        nSens = len(x)
        for _ in range(nSens):
            self.insertHapticSensor(dx=x.pop(), dz=z.pop())


    def insertSampleStructure(self, angle=None):
        pass

    def getSensors(self):
        """return the list of haptic sensors defined"""
        return(self.sensorElements)

    def insertHapticSensor(self, ctr=(0, 4, 18.5), dx=0.0, dz=0.0):
        """insert one single haptic sensor"""
        name = 'haptic' + str(self._nSensorElements)
        self._nSensorElements += 1
        # centers of haptic sensors lie on a cylinder of fixed radius (slightly bigger than fingertip)
        R = 1.2
        alpha = asin(dx / R)
        dy = -R * cos(alpha)
        pos = [dx, dy, dz]
        rot = [90, degrees(alpha), 0]
        # ERP = h kp / (h kp + kd)
        # CFM = 1 / (h kp + kd) = ERP / h kp
        _h = 0.01  # temporal stepwidth
        # CHECKME: unused!
        # for assumed mass of finger of 0.5kg, kp=10, kd=4.5 is approx. the non-oscillatory case
        self.insertBody(name, 'cappedCylinder', [0.08, 0.5], 7, pos=[ctr[0] + pos[0], ctr[1] + pos[1], ctr[2] + pos[2]], \
            euler=rot, passSet=['haptic'])
        jname = 'finger1_' + name
        self.insertJoint('fingertip', name, 'slider', name=jname, \
            axis={'x':-dx, 'y':-dy, 'z':0, "HiStop":0.0, "LowStop":0.0, "StopERP":0.022, "StopCFM":0.22 })
        self.sensorElements.append(jname)

class XODEhand(XODEfile):

    def __init__(self, name, **kwargs):
        """Creates hand with fingertip and palm sensors -- palm up"""
        XODEfile.__init__(self, name, **kwargs)
        # create the hand and finger
        self.insertBody('palm', 'box', [10, 2, 10], 30, pos=[0, 0, 0], passSet=['pal'])
        self.insertBody('pressure', 'box', [8, 0.5, 8], 30, pos=[0, 1, 0], passSet=['pal'])
        self.insertBody('finger0_link0', 'cappedCylinder', [1, 7.5], 5, pos=[-8.75, 0, -2.5], euler=[0, 90, 0], passSet=['pal', 'f01'])
        self.insertBody('finger0_link1', 'cappedCylinder', [1, 4], 5, pos=[-14.5, 0, -2.5], euler=[0, 90, 0], passSet=['f01', 'f02'])
        self.insertBody('finger0_link2', 'cappedCylinder', [1, 2.9], 5, pos=[-17.95, 0, -2.5], euler=[0, 90, 0], passSet=['f02', 'f03'])
        self.insertBody('finger0_link3', 'sphere', [1], 5, pos=[-19, 0, -2.5], passSet=['f03'])
        self.insertBody('finger1_link0', 'cappedCylinder', [1, 7.5], 5, pos=[-3.75, 0, 8.75], passSet=['pal', 'f11'])
        self.insertBody('finger1_link1', 'cappedCylinder', [1, 4], 5, pos=[-3.75, 0, 14.5], passSet=['f11', 'f12'])
        self.insertBody('finger1_link2', 'cappedCylinder', [1, 2.9], 5, pos=[-3.75, 0, 17.95], passSet=['f12', 'f13'])
        self.insertBody('finger1_link3', 'sphere', [1], 5, pos=[-3.75, 0, 19], passSet=['f13'])
        self.insertBody('finger2_link0', 'cappedCylinder', [1, 7.5], 5, pos=[0, 0, 8.75], passSet=['pal', 'f21'])
        self.insertBody('finger2_link1', 'cappedCylinder', [1, 4], 5, pos=[0, 0, 14.5], passSet=['f21', 'f22'])
        self.insertBody('finger2_link2', 'cappedCylinder', [1, 2.9], 5, pos=[0, 0, 17.95], passSet=['f22', 'f23'])
        self.insertBody('finger2_link3', 'sphere', [1], 5, pos=[0, 0, 19], passSet=['f23'])
        self.insertBody('finger3_link0', 'cappedCylinder', [1, 7.5], 5, pos=[3.75, 0, 8.75], passSet=['pal', 'f31'])
        self.insertBody('finger3_link1', 'cappedCylinder', [1, 4], 5, pos=[3.75, 0, 14.5], passSet=['f31', 'f32'])
        self.insertBody('finger3_link2', 'cappedCylinder', [1, 2.9], 5, pos=[3.75, 0, 17.95], passSet=['f32', 'f33'])
        self.insertBody('finger3_link3', 'sphere', [1], 5, pos=[3.75, 0, 19], passSet=['f33'])
        self.insertJoint('palm', 'pressure', 'slider', axis={'x':0, 'y':1, 'z':0, "HiStop":0, "LowStop":-0.5, "StopERP":0.999, "StopCFM":0.002})
        self.insertJoint('palm', 'finger0_link0', 'hinge', axis={'x':0, 'y':0, 'z':1, "HiStop":1.5, "LowStop":0.0}, anchor=(-5, 0, -2.5))
        self.insertJoint('finger0_link0', 'finger0_link1', 'hinge', axis={'x':0, 'y':0, 'z':1, "HiStop":1.5, "LowStop":0.0}, anchor=(-12.5, 0, -2.5))
        self.insertJoint('finger0_link1', 'finger0_link2', 'hinge', axis={'x':0, 'y':0, 'z':1, "HiStop":1.5, "LowStop":0.0}, anchor=(-16.5, 0, -2.5))
        self.insertJoint('finger0_link2', 'finger0_link3', 'slider', axis={'x':1, 'y':0, 'z':0, "HiStop":0, "LowStop":-0.5, "StopERP":0.999, "StopCFM":0.002})
        self.insertJoint('palm', 'finger1_link0', 'hinge', axis={'x':1, 'y':0, 'z':0, "HiStop":1.5, "LowStop":0.0}, anchor=(3.75, 0, 5))
        self.insertJoint('finger1_link0', 'finger1_link1', 'hinge', axis={'x':1, 'y':0, 'z':0, "HiStop":1.5, "LowStop":0.0}, anchor=(3.75, 0, 12.5))
        self.insertJoint('finger1_link1', 'finger1_link2', 'hinge', axis={'x':1, 'y':0, 'z':0, "HiStop":1.5, "LowStop":0.0}, anchor=(3.75, 0, 16.5))
        self.insertJoint('finger1_link2', 'finger1_link3', 'slider', axis={'x':0, 'y':0, 'z':1, "HiStop":0, "LowStop":-0.5, "StopERP":0.999, "StopCFM":0.002})
        self.insertJoint('palm', 'finger2_link0', 'hinge', axis={'x':1, 'y':0, 'z':0, "HiStop":1.5, "LowStop":0.0}, anchor=(0, 0, 5))
        self.insertJoint('finger2_link0', 'finger2_link1', 'hinge', axis={'x':1, 'y':0, 'z':0, "HiStop":1.5, "LowStop":0.0}, anchor=(0, 0, 12.5))
        self.insertJoint('finger2_link1', 'finger2_link2', 'hinge', axis={'x':1, 'y':0, 'z':0, "HiStop":1.5, "LowStop":0.0}, anchor=(0, 0, 16.5))
        self.insertJoint('finger2_link2', 'finger2_link3', 'slider', axis={'x':0, 'y':0, 'z':1, "HiStop":0, "LowStop":-0.5, "StopERP":0.999, "StopCFM":0.002})
        self.insertJoint('palm', 'finger3_link0', 'hinge', axis={'x':1, 'y':0, 'z':0, "HiStop":1.5, "LowStop":0.0}, anchor=(-3.75, 0, 5))
        self.insertJoint('finger3_link0', 'finger3_link1', 'hinge', axis={'x':1, 'y':0, 'z':0, "HiStop":1.5, "LowStop":0.0}, anchor=(-3.75, 0, 12.5))
        self.insertJoint('finger3_link1', 'finger3_link2', 'hinge', axis={'x':1, 'y':0, 'z':0, "HiStop":1.5, "LowStop":0.0}, anchor=(-3.75, 0, 16.5))
        self.insertJoint('finger3_link2', 'finger3_link3', 'slider', axis={'x':0, 'y':0, 'z':1, "HiStop":0, "LowStop":-0.5, "StopERP":0.999, "StopCFM":0.002})
        self.centerOn('palm')
        self.insertFloor(y= -1)
        # add one group of haptic sensors
        self._nSensorElements = 0
        self.sensorElements = []
        self.sensorGroupName = None


class XODEhandflip(XODEfile):

    def __init__(self, name, **kwargs):
        """Creates hand with fingertip and palm sensors -- palm down"""
        XODEfile.__init__(self, name, **kwargs)
        # create the hand and finger
        self.insertBody('palm', 'box', [10, 2, 10], 10, pos=[0, 0, 0], passSet=['pal'])
        self.insertBody('pressure', 'box', [8, 0.5, 8], 10, pos=[0, -1, 0], passSet=['pal'])
        self.insertBody('finger0_link0', 'cappedCylinder', [1, 7.5], 5, pos=[-8.75, 0, -2.5], euler=[0, 90, 0], passSet=['pal', 'f01'])
        self.insertBody('finger0_link1', 'cappedCylinder', [1, 4], 5, pos=[-14.5, 0, -2.5], euler=[0, 90, 0], passSet=['f01', 'f02'])
        self.insertBody('finger0_link2', 'cappedCylinder', [1, 2.9], 5, pos=[-17.95, 0, -2.5], euler=[0, 90, 0], passSet=['f02', 'f03'])
        self.insertBody('finger0_link3', 'sphere', [1], 5, pos=[-19, 0, -2.5], passSet=['f03'])
        self.insertBody('finger1_link0', 'cappedCylinder', [1, 7.5], 5, pos=[-3.75, 0, 8.75], passSet=['pal', 'f11'])
        self.insertBody('finger1_link1', 'cappedCylinder', [1, 4], 5, pos=[-3.75, 0, 14.5], passSet=['f11', 'f12'])
        self.insertBody('finger1_link2', 'cappedCylinder', [1, 2.9], 5, pos=[-3.75, 0, 17.95], passSet=['f12', 'f13'])
        self.insertBody('finger1_link3', 'sphere', [1], 5, pos=[-3.75, 0, 19], passSet=['f13'])
        self.insertBody('finger2_link0', 'cappedCylinder', [1, 7.5], 5, pos=[0, 0, 8.75], passSet=['pal', 'f21'])
        self.insertBody('finger2_link1', 'cappedCylinder', [1, 4], 5, pos=[0, 0, 14.5], passSet=['f21', 'f22'])
        self.insertBody('finger2_link2', 'cappedCylinder', [1, 2.9], 5, pos=[0, 0, 17.95], passSet=['f22', 'f23'])
        self.insertBody('finger2_link3', 'sphere', [1], 5, pos=[0, 0, 19], passSet=['f23'])
        self.insertBody('finger3_link0', 'cappedCylinder', [1, 7.5], 5, pos=[3.75, 0, 8.75], passSet=['pal', 'f31'])
        self.insertBody('finger3_link1', 'cappedCylinder', [1, 4], 5, pos=[3.75, 0, 14.5], passSet=['f31', 'f32'])
        self.insertBody('finger3_link2', 'cappedCylinder', [1, 2.9], 5, pos=[3.75, 0, 17.95], passSet=['f32', 'f33'])
        self.insertBody('finger3_link3', 'sphere', [1], 5, pos=[3.75, 0, 19], passSet=['f33'])
        ## funny finger config with bestNetwork provided (try it ;)
        ##self.insertJoint('palm','pressure','slider', axis={'x':0,'y':-1,'z':0,"HiStop":0,"LowStop":-0.5, "StopERP":0.999,"StopCFM":0.002})
        self.insertJoint('palm', 'pressure', 'slider', axis={'x':0, 'y':1, 'z':0, "HiStop":0, "LowStop":0, "StopERP":0.999, "StopCFM":0.002})
        self.insertJoint('palm', 'finger0_link0', 'hinge', axis={'x':0, 'y':0, 'z':-1, "HiStop":1.5, "LowStop":0.0}, anchor=(-5, 0, -2.5))
        self.insertJoint('finger0_link0', 'finger0_link1', 'hinge', axis={'x':0, 'y':0, 'z':-1, "HiStop":1.5, "LowStop":0.0}, anchor=(-12.5, 0, -2.5))
        self.insertJoint('finger0_link1', 'finger0_link2', 'hinge', axis={'x':0, 'y':0, 'z':-1, "HiStop":1.5, "LowStop":0.0}, anchor=(-16.5, 0, -2.5))
        self.insertJoint('finger0_link2', 'finger0_link3', 'slider', axis={'x':1, 'y':0, 'z':0, "HiStop":0, "LowStop":-0.5, "StopERP":0.999, "StopCFM":0.002})
        self.insertJoint('palm', 'finger1_link0', 'hinge', axis={'x':-1, 'y':0, 'z':0, "HiStop":1.5, "LowStop":0.0}, anchor=(3.75, 0, 5))
        self.insertJoint('finger1_link0', 'finger1_link1', 'hinge', axis={'x':-1, 'y':0, 'z':0, "HiStop":1.5, "LowStop":0.0}, anchor=(3.75, 0, 12.5))
        self.insertJoint('finger1_link1', 'finger1_link2', 'hinge', axis={'x':-1, 'y':0, 'z':0, "HiStop":1.5, "LowStop":0.0}, anchor=(3.75, 0, 16.5))
        self.insertJoint('finger1_link2', 'finger1_link3', 'slider', axis={'x':0, 'y':0, 'z':1, "HiStop":0, "LowStop":-0.5, "StopERP":0.999, "StopCFM":0.002})
        self.insertJoint('palm', 'finger2_link0', 'hinge', axis={'x':-1, 'y':0, 'z':0, "HiStop":1.5, "LowStop":0.0}, anchor=(0, 0, 5))
        self.insertJoint('finger2_link0', 'finger2_link1', 'hinge', axis={'x':-1, 'y':0, 'z':0, "HiStop":1.5, "LowStop":0.0}, anchor=(0, 0, 12.5))
        self.insertJoint('finger2_link1', 'finger2_link2', 'hinge', axis={'x':-1, 'y':0, 'z':0, "HiStop":1.5, "LowStop":0.0}, anchor=(0, 0, 16.5))
        self.insertJoint('finger2_link2', 'finger2_link3', 'slider', axis={'x':0, 'y':0, 'z':1, "HiStop":0, "LowStop":-0.5, "StopERP":0.999, "StopCFM":0.002})
        self.insertJoint('palm', 'finger3_link0', 'hinge', axis={'x':-1, 'y':0, 'z':0, "HiStop":1.5, "LowStop":0.0}, anchor=(-3.75, 0, 5))
        self.insertJoint('finger3_link0', 'finger3_link1', 'hinge', axis={'x':-1, 'y':0, 'z':0, "HiStop":1.5, "LowStop":0.0}, anchor=(-3.75, 0, 12.5))
        self.insertJoint('finger3_link1', 'finger3_link2', 'hinge', axis={'x':-1, 'y':0, 'z':0, "HiStop":1.5, "LowStop":0.0}, anchor=(-3.75, 0, 16.5))
        self.insertJoint('finger3_link2', 'finger3_link3', 'slider', axis={'x':0, 'y':0, 'z':1, "HiStop":0, "LowStop":-0.5, "StopERP":0.999, "StopCFM":0.002})
        self.centerOn('palm')
        self.insertFloor(y= -1.25)
        # add one group of haptic sensors
        self._nSensorElements = 0
        self.sensorElements = []
        self.sensorGroupName = None


class HapticTestSetupWithRidges(XODEfinger):

    def insertSampleStructure(self, angle=30, std=0.05, dist=0.9, **kwargs):
        """create some ridges on the sample"""
        for i in range(16):
            name = 'ridge' + str(i)
            self.insertBody(name, 'cappedCylinder', [0.2, 10], 5, pos=[0, 0.5, random.gauss(15 - dist * i, std)], euler=[0, angle, 0], passSet=['sam'])
            self.insertJoint('sample', name, 'fixed')


class HapticTestSetupWithSpheres(XODEfinger):

    def insertSampleStructure(self, xoffs=0.0, std=0.025, dist=0.9, **kwargs):
        """create four rows of spheres on the sample"""
        dx = [dist * k for k in [-1, 0, 1]]
        dz = [dist * k * 0.5 for k in [0, 1, 0]]
        for i in range(16):
            for k in range(3):
                x = random.gauss(dx[k] + xoffs, std)
                z = random.gauss(15 - dist * i + dz[k], std)
                name = 'sphere' + str(i) + str(k)
                self.insertBody(name, 'sphere', [0.2], 5, pos=[x, 0.5, z], passSet=['sam'])
                self.insertJoint('sample', name, 'fixed')


class HapticTestSetupWithSpirals(XODEfinger):

    def insertSampleStructure(self, std=0.05, xoffs=0.0, dist=1.0, **kwargs):
        """create elongated spiral pattern"""
        rg = 50
        phi = [2.3 + sqrt(f) * pi * 10 / sqrt(rg) for f in range(rg)]
        r = [sqrt(f) * 2.5 / sqrt(rg) for f in range(rg)]
        for k in range(rg):
            x = random.gauss(cos(phi[k]) * r[k], std)
            z = random.gauss(5 + sin(phi[k]) * r[k] * 3, std)
            name = 'sphere' + str(k)
            self.insertBody(name, 'sphere', [0.2], 5, pos=[x, 0.5, z], passSet=['sam'])
            self.insertJoint('sample', name, 'fixed')


class HapticTestSetupWithSine(XODEfinger):

    def insertSampleStructure(self, angle=0, std=0.05, xoffs=0.0, dist=1.0, **kwargs):
        """create rotated sine pattern"""
        rg = 50
        z = [f * 10.0 / rg for f in range(rg)]
        x = [sin(f * 2) * sin(f / 3) * 3.5 for f in z]
        z = [f - 5 for f in z]
        for i in range(rg):
            r = sqrt(x[i] * x[i] + z[i] * z[i])
            if r > 0:
                phi = asin(x[i] / r)
                if z[i] < 0: phi = pi - phi
                phi += radians(angle)
                x[i] = random.gauss(sin(phi) * r, std)
                z[i] = random.gauss(cos(phi) * r, std)
            name = 'sphere' + str(i)
            self.insertBody(name, 'sphere', [0.2], 5, pos=[x[i], 0.5, z[i]], passSet=['sam'])
            self.insertJoint('sample', name, 'fixed')

class XODEJohnnie(XODEfile):

    def __init__(self, name, **kwargs):
        """Creates hand with fingertip and palm sensors -- palm up"""
        XODEfile.__init__(self, name, **kwargs)
        # create the hand and finger
        self.insertBody('palm', 'box', [4.12, 3.0, 2], 30, pos=[0, 0, 0], passSet=['total'], mass=3.356)
        self.insertBody('neck', 'cappedCylinder', [0.25, 5.6], 5, pos=[0, 2.8, 0], euler=[90, 0, 0], passSet=['total'], mass=0.1)
        self.insertBody('head', 'box', [3.0, 1.2, 1.5], 30, pos=[0, 4.0, 0], passSet=['total'], mass=0.1)
        self.insertBody('arm_left', 'cappedCylinder', [0.25, 7.5], 5, pos=[2.06, -2.89, 0], euler=[90, 0, 0], passSet=['total'], mass=2.473)
        self.insertBody('arm_right', 'cappedCylinder', [0.25, 7.5], 5, pos=[-2.06, -2.89, 0], euler=[90, 0, 0], passSet=['total'], mass=2.473)
        self.insertBody('hip', 'cappedCylinder', [0.25, 3.2], 5, pos=[0, -1.6, 0], euler=[90, 0, 0], passSet=['total'], mass=0.192)
        self.insertBody('pelvis', 'cappedCylinder', [0.25, 2.4], 5, pos=[0, -3.2, 0], euler=[0, 90, 0], passSet=['total'], mass=1.0)
        self.insertBody('pelLeft', 'cappedCylinder', [0.25, 0.8], 5, pos=[1.2, -3.6, 0], euler=[90, 0, 0], passSet=['total'], mass=2.567)
        self.insertBody('pelRight', 'cappedCylinder', [0.25, 0.8], 5, pos=[-1.2, -3.6, 0], euler=[90, 0, 0], passSet=['total'], mass=2.567)
        self.insertBody('tibiaLeft', 'cappedCylinder', [0.25, 4.4], 5, pos=[1.2, -6.2, 0], euler=[90, 0, 0], passSet=['total'], mass=5.024)
        self.insertBody('tibiaRight', 'cappedCylinder', [0.25, 4.4], 5, pos=[-1.2, -6.2, 0], euler=[90, 0, 0], passSet=['total'], mass=5.024)
        self.insertBody('sheenLeft', 'cappedCylinder', [0.25, 3.8], 5, pos=[1.2, -10.3, 0], euler=[90, 0, 0], passSet=['total'], mass=3.236)
        self.insertBody('sheenRight', 'cappedCylinder', [0.25, 3.8], 5, pos=[-1.2, -10.3, 0], euler=[90, 0, 0], passSet=['total'], mass=3.236)
        self.insertBody('footLeft', 'box', [2.2, 0.4, 2.6], 3, pos=[1.2, -12.2, 0.75], passSet=['total'], mass=1.801)
        self.insertBody('footRight', 'box', [2.2, 0.4, 2.6], 3, pos=[-1.2, -12.2, 0.75], passSet=['total'], mass=1.801)
        self.insertJoint('palm', 'neck', 'fixed', axis={'x':0, 'y':0, 'z':0}, anchor=(0, 0, 0))
        self.insertJoint('neck', 'head', 'fixed', axis={'x':0, 'y':0, 'z':0}, anchor=(0, 2.8, 0))
        self.insertJoint('palm', 'arm_left', 'hinge', axis={'x':1, 'y':0, 'z':0}, anchor=(2.06, 0.86, 0))
        self.insertJoint('palm', 'arm_right', 'hinge', axis={'x':1, 'y':0, 'z':0}, anchor=(-2.06, 0.86, 0))
        self.insertJoint('palm', 'hip', 'hinge', axis={'x':0, 'y':1, 'z':0, "HiStop":0.5, "LowStop":-0.5}, anchor=(0, -1.6, 0))
        self.insertJoint('hip', 'pelvis', 'fixed', axis={'x':0, 'y':0, 'z':0}, anchor=(0, -3.2, 0))
        self.insertJoint('pelvis', 'pelLeft', 'hinge', axis={'x':0, 'y':0, 'z':-1, "HiStop":0.5, "LowStop":0.0}, anchor=(1.2, -3.2, 0))
        self.insertJoint('pelvis', 'pelRight', 'hinge', axis={'x':0, 'y':0, 'z':1, "HiStop":0.5, "LowStop":0.0}, anchor=(-1.2, -3.2, 0))
        self.insertJoint('pelLeft', 'tibiaLeft', 'hinge', axis={'x':1, 'y':0, 'z':0, "HiStop":1.5, "LowStop":0.0}, anchor=(1.2, -4.0, 0))
        self.insertJoint('pelRight', 'tibiaRight', 'hinge', axis={'x':1, 'y':0, 'z':0, "HiStop":1.5, "LowStop":0.0}, anchor=(-1.2, -4.0, 0))
        self.insertJoint('tibiaLeft', 'sheenLeft', 'hinge', axis={'x':-1, 'y':0, 'z':0, "HiStop":1.5, "LowStop":0.0}, anchor=(1.2, -8.4, 0))
        self.insertJoint('tibiaRight', 'sheenRight', 'hinge', axis={'x':-1, 'y':0, 'z':0, "HiStop":1.5, "LowStop":0.0}, anchor=(-1.2, -8.4, 0))
        self.insertJoint('sheenLeft', 'footLeft', 'hinge', axis={'x':1, 'y':0, 'z':0, "HiStop":0.25, "LowStop":-0.25}, anchor=(1.2, -12.2, 0))
        self.insertJoint('sheenRight', 'footRight', 'hinge', axis={'x':1, 'y':0, 'z':0, "HiStop":0.25, "LowStop":-0.25}, anchor=(-1.2, -12.2, 0))
        self.centerOn('palm')
        self.insertFloor(y= -12.7)
        # add one group of haptic sensors
        self._nSensorElements = 0
        self.sensorElements = []
        self.sensorGroupName = None

class XODESLR(XODEfile):
    def __init__(self, name, **kwargs):
        """Creates hand with fingertip and palm sensors -- palm up"""
        XODEfile.__init__(self, name, **kwargs)
        # create the hand and finger
        self.insertBody('body', 'box', [7.0, 16.0, 10.0], 30, pos=[0, 0, 2.0], passSet=['total'], mass=15.0, color=(0.5, 0.5, 0.4, 1.0))
        #right arm
        self.insertBody('shoulderUpRight', 'cappedCylinder', [0.5, 2.0], 5, pos=[2.5, 7.0, -3.5], euler=[0, 90, 0], passSet=['rightSh', 'total'], mass=0.25)
        self.insertBody('shoulderLRRight', 'cappedCylinder', [0.5, 2.0], 5, pos=[3.5, 6.0, -3.5], euler=[90, 0, 0], passSet=['rightSh'], mass=0.25)
        self.insertBody('shoulderPRRight', 'cappedCylinder', [0.5, 2.0], 5, pos=[3.5, 4.0, -3.5], euler=[90, 0, 0], passSet=['rightSh'], mass=0.25)
        self.insertBody('armUpRight', 'cappedCylinder', [0.5, 2.0], 5, pos=[3.5, 2.0, -3.5], euler=[90, 0, 0], passSet=['rightAr', 'rightSh'], mass=0.25)
        self.insertBody('armPRRight', 'cappedCylinder', [0.5, 2.0], 5, pos=[3.5, 0.0, -3.5], euler=[90, 0, 0], passSet=['rightAr'], mass=0.25)
        self.insertBody('handUpRight', 'cappedCylinder', [0.5, 2.0], 5, pos=[3.5, -2.0, -3.5], euler=[90, 0, 0], passSet=['rightAr', 'rightHa'], mass=0.25)
        #right hand
        self.insertBody('palmRight', 'box', [1.5, 0.25, 0.5], 30, pos=[3.5, -3.0, -3.5], passSet=['rightHa'], mass=0.1, color=(0.6, 0.6, 0.3, 1.0))
        self.insertBody('fingerRight1', 'box', [0.25, 1.0, 0.5], 30, pos=[4.0, -3.5, -3.5], passSet=['rightHa'], mass=0.1, color=(0.6, 0.6, 0.3, 1.0))
        self.insertBody('fingerRight2', 'box', [0.25, 1.0, 0.5], 30, pos=[3.0, -3.5, -3.5], passSet=['rightHa'], mass=0.1, color=(0.6, 0.6, 0.3, 1.0))
        
        #left arm
        self.insertBody('shoulderUpLeft', 'cappedCylinder', [0.5, 2.0], 5, pos=[-2.5, 7.0, -3.5], euler=[0, 90, 0], passSet=['leftSh', 'total'], mass=0.25)
        self.insertBody('shoulderLRLeft', 'cappedCylinder', [0.5, 2.0], 5, pos=[-3.5, 6.0, -3.5], euler=[90, 0, 0], passSet=['leftSh'], mass=0.25)
        self.insertBody('shoulderPRLeft', 'cappedCylinder', [0.5, 2.0], 5, pos=[-3.5, 4.0, -3.5], euler=[90, 0, 0], passSet=['leftSh'], mass=0.25)
        self.insertBody('armUpLeft', 'cappedCylinder', [0.5, 2.0], 5, pos=[-3.5, 2.0, -3.5], euler=[90, 0, 0], passSet=['leftAr', 'leftSh'], mass=0.25)
        self.insertBody('armPRLeft', 'cappedCylinder', [0.5, 2.0], 5, pos=[-3.5, 0.0, -3.5], euler=[90, 0, 0], passSet=['leftAr'], mass=0.25)
        self.insertBody('handUpLeft', 'cappedCylinder', [0.5, 2.0], 5, pos=[-3.5, -2.0, -3.5], euler=[90, 0, 0], passSet=['leftAr', 'leftHa'], mass=0.25)
        #left hand
        self.insertBody('palmLeft', 'box', [1.5, 0.25, 0.5], 30, pos=[-3.5, -3.0, -3.5], passSet=['leftHa'], mass=0.1, color=(0.6, 0.6, 0.3, 1.0))
        self.insertBody('fingerLeft1', 'box', [0.25, 1.0, 0.5], 30, pos=[-4.0, -3.5, -3.5], passSet=['leftHa'], mass=0.1, color=(0.6, 0.6, 0.3, 1.0))
        self.insertBody('fingerLeft2', 'box', [0.25, 1.0, 0.5], 30, pos=[-3.0, -3.5, -3.5], passSet=['leftHa'], mass=0.1, color=(0.6, 0.6, 0.3, 1.0))
        
        #Joints right        
        self.insertJoint('body', 'shoulderUpRight', 'hinge', axis={'x':1, 'y':0, 'z':0}, anchor=(2.5, 7.0, -3.5))
        self.insertJoint('shoulderUpRight', 'shoulderLRRight', 'hinge', axis={'x':0, 'y':0, 'z':1}, anchor=(3.5, 7.0, -3.5))
        self.insertJoint('shoulderLRRight', 'shoulderPRRight', 'hinge', axis={'x':0, 'y':1, 'z':0}, anchor=(3.5, 5.0, -3.5))
        self.insertJoint('shoulderPRRight', 'armUpRight', 'hinge', axis={'x':1, 'y':0, 'z':0}, anchor=(3.5, 3.0, -3.5))
        self.insertJoint('armUpRight', 'armPRRight', 'hinge', axis={'x':0, 'y':1, 'z':0}, anchor=(3.5, 1.0, -3.5))
        self.insertJoint('armPRRight', 'handUpRight', 'hinge', axis={'x':1, 'y':0, 'z':0}, anchor=(3.5, -1.0, -3.5))
        self.insertJoint('handUpRight', 'palmRight', 'hinge', axis={'x':0, 'y':1, 'z':0}, anchor=(3.5, -3.0, -3.5))
        self.insertJoint('palmRight', 'fingerRight1', 'fixed', axis={'x':0, 'y':0, 'z':0}, anchor=(4.0, -3.5, -3.5))
        self.insertJoint('palmRight', 'fingerRight2', 'hinge', axis={'x':0, 'y':0, 'z':1}, anchor=(3.0, -3.0, -3.5))
        
        #Joints left        
        self.insertJoint('body', 'shoulderUpLeft', 'hinge', axis={'x':1, 'y':0, 'z':0}, anchor=(-2.5, 7.0, -3.5))
        self.insertJoint('shoulderUpLeft', 'shoulderLRLeft', 'hinge', axis={'x':0, 'y':0, 'z':1}, anchor=(-3.5, 7.0, -3.5))
        self.insertJoint('shoulderLRLeft', 'shoulderPRLeft', 'hinge', axis={'x':0, 'y':1, 'z':0}, anchor=(-3.5, 5.0, -3.5))
        self.insertJoint('shoulderPRLeft', 'armUpLeft', 'hinge', axis={'x':1, 'y':0, 'z':0}, anchor=(-3.5, 3.0, -3.5))
        self.insertJoint('armUpLeft', 'armPRLeft', 'hinge', axis={'x':0, 'y':1, 'z':0}, anchor=(-3.5, 1.0, -3.5))
        self.insertJoint('armPRLeft', 'handUpLeft', 'hinge', axis={'x':1, 'y':0, 'z':0}, anchor=(-3.5, -1.0, -3.5))
        self.insertJoint('handUpLeft', 'palmLeft', 'hinge', axis={'x':0, 'y':1, 'z':0}, anchor=(-3.5, -3.0, -3.5))
        self.insertJoint('palmLeft', 'fingerLeft1', 'fixed', axis={'x':0, 'y':0, 'z':0}, anchor=(-4.0, -3.5, -3.5))
        self.insertJoint('palmLeft', 'fingerLeft2', 'hinge', axis={'x':0, 'y':0, 'z':1}, anchor=(-3.0, -3.0, -3.5))
               
        self.centerOn('body')
        self.insertFloor(y= -8.0)
        # add one group of haptic sensors
        self._nSensorElements = 0
        self.sensorElements = []
        self.sensorGroupName = None

class XODELSRTable(XODESLR): #XODESLR
    def __init__(self, name, **kwargs):
        XODESLR.__init__(self, name, **kwargs)        
        # create table
        self.insertBody('plate', 'box', [15.0, 1.0, 8.0], 30, pos=[-12.5, 0.5, -14.0], passSet=['table'], mass=2.0, color=(0.4, 0.25, 0.0, 1.0))
        self.insertBody('leg1', 'box', [0.5, 8.0, 0.5], 30, pos=[-19.5, -4.0, -17.5], passSet=['table'], mass=0.3, color=(0.6, 0.8, 0.8, 0.8))
        self.insertJoint('plate', 'leg1', 'fixed', axis={'x':0, 'y':0, 'z':0}, anchor=(-19.5, 0.0, -17.5))
        self.insertBody('leg2', 'box', [0.5, 8.0, 0.5], 30, pos=[-5.5, -4.0, -17.5], passSet=['table'], mass=0.3, color=(0.6, 0.8, 0.8, 0.8))
        self.insertJoint('plate', 'leg2', 'fixed', axis={'x':0, 'y':0, 'z':0}, anchor=(-5.5, 0.0, -17.5))
        self.insertBody('leg3', 'box', [0.5, 8.0, 0.5], 30, pos=[-5.5, -4.0, -10.5], passSet=['table'], mass=0.3, color=(0.6, 0.8, 0.8, 0.8))
        self.insertJoint('plate', 'leg3', 'fixed', axis={'x':0, 'y':0, 'z':0}, anchor=(-5.5, 0.0, -10.5))
        self.insertBody('leg4', 'box', [0.5, 8.0, 0.5], 30, pos=[-19.5, -4.0, -10.5], passSet=['table'], mass=0.3, color=(0.6, 0.8, 0.8, 0.8))
        self.insertJoint('plate', 'leg4', 'fixed', axis={'x':0, 'y':0, 'z':0}, anchor=(-19.5, 0.0, -10.5))

class XODELSRGlas(XODELSRTable): #XODESLR
    def __init__(self, name, **kwargs):
        XODELSRTable.__init__(self, name, **kwargs)
        # create glass + coaster (necessary because cylinder collision has a bug)
        self.insertBody('objectP00', 'cylinder', [0.2, 1], 30, pos=[-6.5, 1.51 , -11.0], passSet=['object'], mass=0.2, euler=[90, 0, 0], color=(0.6, 0.6, 0.8, 0.5))
        self.insertBody('objectP01', 'box', [0.45, 0.02, 0.45], 30, pos=[-6.5, 1.01, -11.0], passSet=['object'], mass=0.01)
        self.insertBody('objectP02', 'box', [0.45, 0.02, 0.45], 30, pos=[-6.5, 2.01, -11.0], passSet=['object'], mass=0.01)
        self.insertJoint('objectP00', 'objectP01', 'fixed', axis={'x':0, 'y':0, 'z':0}, anchor=(-6.5, 1.01, -11.0))
        self.insertJoint('objectP00', 'objectP02', 'fixed', axis={'x':0, 'y':0, 'z':0}, anchor=(-6.5, 2.01, -11.0))

class XODELSRPlate(XODELSRTable): #XODESLR
    def __init__(self, name, **kwargs):
        XODELSRTable.__init__(self, name, **kwargs)
        # create plate
        # plate ground
        bX = 1.0 #width of plate floor
        bY = 0.05 #height of plate floor
        bZ = 1.0 #depth of plate floor
        #plate sides
        sX = 0.5 #width of plate side
        sY = bY #height of plate side
        sZ = 1.0 #depth of plate side
        #position of plate
        pX = -6.5
        pY = 1.02
        pZ = -11.0
        #stuff
        m = 0.05 #mass per part
        c = (0.6, 0.6, 0.8, 0.95) #color of object
        dif = sX / (2.0 * sqrt(5)) #

        self.insertBody('objectP00', 'box', [bX, bY, bZ], 30, pos=[pX, pY, pZ], passSet=['object'], mass=m, color=c)
        self.insertBody('objectP01', 'box', [sX, sY, sZ], 30, pos=[pX - bX * 0.5 - 2.0 * dif, pY + dif, pZ], passSet=['object'], mass=m, euler=[0, 0, 22.5], color=c)
        self.insertJoint('objectP00', 'objectP01', 'fixed', axis={'x':0, 'y':0, 'z':0}, anchor=(pX - bX * 0.5, pY, pZ))

        self.insertBody('objectP02', 'box', [sX, sY, sZ], 30, pos=[pX + bX * 0.5 + 2.0 * dif, pY + dif, pZ], passSet=['object'], mass=m, euler=[0, 0, -22.5], color=c)
        self.insertJoint('objectP00', 'objectP02', 'fixed', axis={'x':0, 'y':0, 'z':0}, anchor=(pX + bX * 0.5, pY, pZ))

        self.insertBody('objectP03', 'box', [sX, sY, sZ], 30, pos=[pX, pY + dif, pZ + bZ * 0.5 + 2.0 * dif], passSet=['object'], mass=m, euler=[0, 90, -22.5], color=c)
        self.insertJoint('objectP00', 'objectP03', 'fixed', axis={'x':0, 'y':0, 'z':0}, anchor=(pX, pY, pZ + bZ * 0.5))

        self.insertBody('objectP04', 'box', [sX, sY, sZ], 30, pos=[pX, pY + dif, pZ - bZ * 0.5 - 2.0 * dif], passSet=['object'], mass=m, euler=[0, 90, 22.5], color=c)
        self.insertJoint('objectP00', 'objectP04', 'fixed', axis={'x':0, 'y':0, 'z':0}, anchor=(pX, pY, pZ - bZ * 0.5))
        
if __name__ == '__main__' :

    table = XODELSRPlate('../models/ccrlPlate')
    
    #z = XODESLR('../models/slr')
    #z = XODEhand('hand_mal_10')
    #z = XODEhandflip('handflip')
    #z = XODEhandflip('handflip')
    #z.scaleModel(0.5)
    
    table.writeXODE()

