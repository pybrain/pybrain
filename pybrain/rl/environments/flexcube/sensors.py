__author__ = 'Frank Sehnke, sehnke@in.tum.de'

#########################################################################
# The sensors that are available for the FlexCube Environment
#
# The FlexCube Environment is a Mass-Spring-System composed of 8 mass points.
# These resemble a cube with flexible edges.
#
# A wide variety of sensors are available for observation and reward:
# - EdgesReal: 12 edge lengths
# - EdgesTarget: 12 wanted edge lengths (the last action)
# - VerticesContact: vertexes contact with floor
# - VerticesMinHight: distance of closest vertex to the floor
# - DistToOrigin: distance to origin (0.0, 0.0)
# - Target: distance and angle to target
# - Time: time dependend cyclic signals
#
#########################################################################

from scipy import sqrt, zeros, array, clip, sin

# Class that bundles the different sensors defined by their names in sensorList
class Sensors:
    def __init__(self, sensorList):
        self.sensors = []
        for i in sensorList:
            self.sensors.append(eval(i + "()"))

    def updateSensor(self, pos, vel, dist, center, step, wEdges):
        for i in self.sensors:
            i.updateSensor(pos, vel, dist, center, step, wEdges)

    def getSensor(self):
        output = []
        for i in self.sensors:
            output.append(i.getSensor()[:])
        return output

#Sensor basis class
class defaultSensor:
    def __init__(self):
        self.sensorOutput = ["defaultSensor", 0]
        self.targetList = [array([-80.0, 0.0, 0.0])]
        self.edges = array([1, 2, 4, 11, 13, 19, 22, 31, 37, 38, 47, 55])

    def updateSensor(self, pos, vel, dist, center, step, wEdges):
        self.pos = pos.copy()
        self.dist = dist.copy()
        self.centerOfGrav = center.copy().reshape(3)
        self.centerOfGrav[1] = 0.0
        self.step = step
        self.wantedEdges = wEdges.copy()

    def getSensor(self):
        return self.sensorOutput

#Gives back the length of the edges of the cube
class EdgesReal(defaultSensor):
    def getSensor(self):
        self.sensorOutput = ["EdgesReal", 12]
        self.sensorOutput.append(self.dist[self.edges].reshape(12) / 30.0) #/30 for normalization [0-1]
        return self.sensorOutput

#Rewardsensor for grow task - returns how much the sum of edge lengths exeeds initial sum
class EdgesSumReal(defaultSensor):
    def getSensor(self):
        self.sensorOutput = ["EdgesSumReal", 1]
        self.sensorOutput.append(array([(self.dist[self.edges].reshape(12)).sum(axis=0) - 240.0]))
        return self.sensorOutput

#Gives back the desired length of the edges (last action)
class EdgesTarget(defaultSensor):
    def getSensor(self):
        self.sensorOutput = ["EdgesTarget", 12]
        self.sensorOutput.append(self.wantedEdges.reshape(12) / 30.0)
        return self.sensorOutput

#Returns how close vertices are to the floor up to 1 pixel distance (contact sensor)
class VerticesContact(defaultSensor):
    def getSensor(self):
        self.sensorOutput = ["VerticesContact", 8]
        self.sensorOutput.append(clip((1.0 - self.pos[:, 1]), 0.0, 1.0))
        return self.sensorOutput

#Returns the distance of the closest vertex to the floor (reward for jump task)
class VerticesMinHight(defaultSensor):
    def getSensor(self):
        self.sensorOutput = ["VerticesMinHight", 1]
        self.sensorOutput.append(array([min(self.pos[:, 1])]))
        return self.sensorOutput

#Returns the distance to the origin of the cube (reward for walk task)
class DistToOrigin(defaultSensor):
    def getSensor(self):
        self.sensorOutput = ["DistToOrigin", 1]
        self.sensorOutput.append(array([sqrt((self.centerOfGrav ** 2).sum(axis=0)) * 0.00125]))
        return self.sensorOutput

#Returns the distance and angle to a target point (distance is reward for target task)
class Target(defaultSensor):
    def getSensor(self):
        self.sensorOutput = ["Target", 5]

        t = self.targetList[0] - self.centerOfGrav
        dist = sqrt((t ** 2).sum(axis=0))
        out = zeros(5, float)
        out[0] = dist * 0.00625

        for i in range(4):
            if i < 2:
                d = self.pos[i] - self.centerOfGrav
            else:
                d = self.pos[i + 2] - self.centerOfGrav
            sen = sqrt((d ** 2).sum(axis=0))
            norm = dist * sen
            cosA = (d[0] * t[0] + d[2] * t[2]) / norm
            sinA = (d[0] * t[2] - d[2] * t[0]) / norm
            if cosA < 0.0:
                if sinA > 0.0: sinA = 1.0
                else: sinA = -1.0
            out[i + 1] = sinA
        self.sensorOutput.append(out)
        return self.sensorOutput

#Returns 3 time dependend values given by the sin of the timestep in the current episode
class Time(defaultSensor):
    def getSensor(self):
        self.sensorOutput = ["Time", 3]
        self.sensorOutput.append(array([sin(float(self.step) / 4.0), sin(float(self.step) / 8.0), sin(float(self.step) / 16.0)]))
        return self.sensorOutput

