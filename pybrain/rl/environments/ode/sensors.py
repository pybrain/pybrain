__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

import ode, sys, xode #@UnresolvedImport
import warnings
from scipy.linalg import norm
from pybrain.utilities import Named

class SizeError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return 'size not correct: ', repr(self.value)


class Sensor(Named):
    """The base Sensor class. Every sensor has a name, and a list of values (even if it is
        only one value) with numValues entries. They can be added to the ODEEnvironment with
        its addSensor(...) function. Sensors get the world model when added to the world.
        Other information sources have to be taken manually by parsing the world object."""

    def __init__(self, name, numValues):
        self._numValues = numValues
        self.name = name
        self._values = [0] * self._numValues
        self._world = None

    def _update(self):
        pass

    def _connect(self, world):
        self._world = world

    def setNumValues(self, numValues):
        self._numValues = numValues

    def getNumValues(self):
        return self._numValues

    def setValues(self, values):
        if (len(values) != self._numValues):
            raise SizeError(self._numValues)
        self._values = values[:]

    def getValues(self):
        return self._values



class JointSensor(Sensor):
    ''' This sensor parses the xode root node for all joints and returns the angles
        for each of them. Different joints have a different number of values (e.g.
        a hinge2 joints has two degrees of freedom, whereas a slider joint has only one).
        However, calling the function getValues(), will return a flat list of all the
        degrees of freedom of all joints.'''

    def __init__(self, name='JointSensor'):
        Sensor.__init__(self, name, 0)
        self._joints = []

    def _parseJoints(self, node):
        if isinstance(node, xode.joint.Joint):
            # append joints to joint vector
            joint = node.getODEObject()
            joint.name = node.getName()
            self._joints.append(joint)
        # recursive call for children
        for c in node.getChildren():
            self._parseJoints(c)

    def _connect(self, world):
        Sensor._connect(self, world)

        # get XODE Root and parse its joints
        self._joints = []
        self._parseJoints(self._world.getXODERoot())

        # do initial update to get numValues
        self._update()
        self._numValues = len(self._values)

    def _update(self):
        self._values = []
        for j in self._joints:
            if type(j) == ode.BallJoint:
                # ball joints can't be controlled yet
                pass
            elif type(j) == ode.UniversalJoint:
                # universal joints not implemented (are covered by using AMotors instead
                pass
            elif type(j) == ode.AMotor:
                num = j.getNumAxes()
                for i in range(num):
                    self._values.append(j.getAngle(i))
            elif type(j) == ode.HingeJoint:
                # a hinge joint has only one angle
                self._values.append(j.getAngle())
            elif type(j) == ode.Hinge2Joint:
                # for hinge2 joints, we need to values
                self._values.append(j.getAngle1())
                self._values.append(j.getAngle2())
            elif type(j) == ode.SliderJoint:
                # slider joints have one value (the relative distance)
                self._values.append(j.getPosition())

    def getJoints(self):
        return self._joints


class JointVelocitySensor(JointSensor):

    def __init__(self, name='JointVelocitySensor'):
        Sensor.__init__(self, name, 0)
        self._joints = []

    def _update(self):
        self._values = []
        for j in self._joints:
            if type(j) == ode.BallJoint:
                # ball joints can't be controlled yet
                pass
            elif type(j) == ode.UniversalJoint:
                # universal joints not implemented (are covered by using AMotors instead
                pass
            elif type(j) == ode.AMotor:
                num = j.getNumAxes()
                for i in range(num):
                    self._values.append(j.getAngleRate(i))
            elif type(j) == ode.HingeJoint:
                # a hinge joint has only one angle
                self._values.append(j.getAngleRate())
            elif type(j) == ode.Hinge2Joint:
                # for hinge2 joints, we need to values
                self._values.append(j.getAngle1Rate())
                self._values.append(j.getAngle2Rate())
            elif type(j) == ode.SliderJoint:
                # slider joints have one value (the relative distance)
                self._values.append(j.getPositionRate())


class DistToPointSensor(Sensor):
    ''' This sensor takes a name (of a body) and possibly a point and returns
        the current distance of the body to this point. if no point is given,
        the distance to the origin is returned. '''

    def __init__(self, bodyName, name='DistToPointSensor', point=(0, 0, 0)):
        Sensor.__init__(self, name, 0)
        # initialize one return value
        self.setNumValues(1)
        self._values = [0]
        self._bodyName = bodyName
        self._point = point

    def _update(self):
        try:
            odeObj = self._world.getXODERoot().namedChild(self._bodyName).getODEObject()
        except KeyError:
            # the given object name is not found. output warning and return distance 0.
            warnings.warn("Object with name '", self._bodyName, "' not found.")
            return 0
        self.setValues([norm(odeObj.getPosRelPoint(self._point))])

class BodyDistanceSensor(Sensor):
    ''' This sensor takes two body names and returns the current distance between them. '''

    def __init__(self, bodyName1, bodyName2, name='BodyDistanceSensor'):
        Sensor.__init__(self, name, 0)
        # initialize one return value
        self.setNumValues(1)
        self._values = [0]
        self._bodyName1 = bodyName1
        self._bodyName2 = bodyName2

    def _update(self):
        try:
            odeObj1 = self._world.getXODERoot().namedChild(self._bodyName1).getODEObject()
            odeObj2 = self._world.getXODERoot().namedChild(self._bodyName2).getODEObject()
        except KeyError:
            # the given object name is not found. output warning and return distance 0.
            warnings.warn("One of the objects '", self._bodyName1, self._bodyName2, "' was not found.")
            return 0
        self.setValues([norm(odeObj1.getPosRelPoint(odeObj2.getPosition()))])


class BodyPositionSensor(Sensor):
    ''' This sensor parses the xode root node for all bodies and returns the positions in x, y, z
        for each of them. Calling the function getValues(), will return a flat list of all the
        bodies' coordinates.'''

    def __init__(self, name='BodyPositionSensor'):
        Sensor.__init__(self, name, 0)
        self._bodies = []

    def _parseBodies(self, node):
        if isinstance(node, xode.body.Body):
            # append bodies to body vector
            body = node.getODEObject()
            body.name = node.getName()
            self._bodies.append(body)
        # recursive call for children
        for c in node.getChildren():
            self._parseBodies(c)

    def _connect(self, world):
        Sensor._connect(self, world)

        # get XODE Root and parse its joints
        self._bodies = []
        self._parseBodies(self._world.getXODERoot())

        # do initial update to get numValues
        self._update()
        self._numValues = len(self._values)

    def _update(self):
        self._values = []
        for b in self._bodies:
            self._values.extend(b.getPosition())

    def getBodies(self):
        return self._bodies


class SpecificJointSensor(JointSensor):
    ''' This sensor takes a list of joint names, and returns only their values. '''

    def __init__(self, jointNames, name=None):
        Sensor.__init__(self, name, 0)
        self._names = jointNames
        self._joints = []
        self._values = []

    def _parseJoints(self, node=None):
        for name in self._names:
            try:
                self._joints.append(self._world.getXODERoot().namedChild(name).getODEObject())
            except KeyError:
                # the given object name is not found. output warning and quit.
                warnings.warn("Joint with name '", name, "' not found.")
                sys.exit()

    def _connect(self, world):
        """ Connects the sensor to the world and initializes the value list. """
        Sensor._connect(self, world)

        # initialize object list - this should not change during runtime
        self._joints = []
        self._parseJoints()

        # do initial update to get numValues
        self._update()
        self._numValues = len(self._values)


class SpecificJointVelocitySensor(JointVelocitySensor):
    ''' This sensor takes a list of joint names, and returns only their velocities. '''

    def __init__(self, jointNames, name=None):
        Sensor.__init__(self, name, 0)
        self._names = jointNames
        self._joints = []
        self._values = []

    def _parseJoints(self, node=None):
        for name in self._names:
            try:
                self._joints.append(self._world.getXODERoot().namedChild(name).getODEObject())
            except KeyError:
                # the given object name is not found. output warning and quit.
                warnings.warn("Joint with name '", name, "' not found.")
                sys.exit()

    def _connect(self, world):
        """ Connects the sensor to the world and initializes the value list. """
        Sensor._connect(self, world)

        # initialize object list - this should not change during runtime
        self._joints = []
        self._parseJoints()

        # do initial update to get numValues
        self._update()
        self._numValues = len(self._values)


class SpecificBodyPositionSensor(BodyPositionSensor):
    ''' This sensor takes a list of body names, and returns their positions. It must
        be given a custom name as well, for later identification.'''

    def __init__(self, bodyNames, name=None):
        Sensor.__init__(self, name, 0)
        self._names = bodyNames
        self._bodies = []
        self._values = [0]

    def _parseBodies(self, node=None):
        for name in self._names:
            try:
                self._bodies.append(self._world.getXODERoot().namedChild(name).getODEObject())
            except KeyError:
                # the given object name is not found. output warning and quit.
                warnings.warn("Body with name '", name, "' not found.")
                sys.exit()

    def _connect(self, world):
        """ Connects the sensor to the world and initializes the value list. """
        Sensor._connect(self, world)

        # initialize object list - this should not change during runtime
        self._bodies = []
        self._parseBodies()

        # do initial update to get numValues
        self._update()
        self._numValues = len(self._values)


