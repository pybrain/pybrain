__author__ = 'Frank Sehnke, sehnke@in.tum.de'

import sensors
import threading
from pybrain.utilities import threaded
from pybrain.tools.networking.udpconnection import UDPServer
from pybrain.rl.environments.environment import Environment
from scipy import ones, zeros, array, clip, arange, sqrt
from time import sleep

class FlexCubeEnvironment(Environment):
    def __init__(self, render=True, realtime=True, ip="127.0.0.1", port="21560"):
        # initialize base class
        self.render = render
        if self.render:
            self.updateDone = True
            self.updateLock = threading.Lock()
            self.server = UDPServer(ip, port)
        self.actLen = 12
        self.mySensors = sensors.Sensors(["EdgesReal"])
        self.dists = array([20.0, sqrt(2.0) * 20, sqrt(3.0) * 20])
        self.gravVect = array([0.0, -100.0, 0.0])
        self.centerOfGrav = zeros((1, 3), float)
        self.pos = ones((8, 3), float)
        self.vel = zeros((8, 3), float)
        self.SpringM = ones((8, 8), float)
        self.d = 60.0
        self.dt = 0.02
        self.startHight = 10.0
        self.dumping = 0.4
        self.fraktMin = 0.7
        self.fraktMax = 1.3
        self.minAkt = self.dists[0] * self.fraktMin
        self.maxAkt = self.dists[0] * self.fraktMax
        self.reset()
        self.count = 0
        self.setEdges()
        self.act(array([20.0] * 12))
        self.euler()
        self.realtime = realtime
        self.step = 0

    def closeSocket(self):
        self.server.UDPInSock.close()
        sleep(10)

    def setEdges(self):
        self.edges = zeros((12, 2), int)
        count = 0
        c1 = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    c2 = 0
                    for i2 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                sum = abs(i - i2) + abs(j - j2) + abs(k - k2)
                                if sum == 1 and i <= i2 and j <= j2 and k <= k2:
                                    self.edges[count] = [c1, c2]
                                    count += 1
                                c2 += 1
                    c1 += 1

    def reset(self):
        self.action = ones((1, 12), float) * self.dists[0]

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    self.pos[i * 4 + j * 2 + k] = [i * self.dists[0] - self.dists[0] / 2.0, j * self.dists[0] - self.dists[0] / 2.0 + self.startHight, k * self.dists[0] - self.dists[0] / 2.0]
        self.vel = zeros((8, 3), float)

        idx0 = arange(8).repeat(8)
        idx1 = array(range(8) * 8)
        self.difM = self.pos[idx0, :] - self.pos[idx1, :] #vectors from all points to all other points
        self.springM = sqrt((self.difM ** 2).sum(axis=1)).reshape(64, 1)
        self.distM = self.springM.copy() #distance matrix
        self.step = 0
        self.mySensors.updateSensor(self.pos, self.vel, self.distM, self.centerOfGrav, self.step, self.action)
        if self.render:
            if self.server.clients > 0:
                # If there are clients send them reset signal
                self.server.send(["r", "r"])

    def performAction(self, action):
        action = self.normAct(action)
        self.action = action.copy()
        self.act(action)
        self.euler()
        self.step += 1

        if self.render:
            if self.updateDone:
                self.updateRenderer()
                if self.server.clients > 0 and self.realtime:
                    sleep(0.02)

    def getSensors(self):
        self.mySensors.updateSensor(self.pos, self.vel, self.distM, self.centerOfGrav, self.step, self.action)
        return self.mySensors.getSensor()[:]

    def normAct(self, s):
        return clip(s, self.minAkt, self.maxAkt)

    def act(self, a):
        count = 0
        for i in self.edges:
            self.springM[i[0] * 8 + i[1]] = a[count]
            self.springM[i[1] * 8 + i[0]] = a[count]
            count += 1

    def euler(self):
        self.count += 1
        #Inner Forces
        distM = self.distM.copy()
        disM = self.springM - distM #difference between wanted spring lengths and current ones
        disM = disM.reshape(64, 1)

        distM = distM + 0.0000000001 #hack to prevent divs by 0

        #Forces to Velos
        #spring vectors normalized to 1 times the actual force from deformation
        vel = self.difM / distM
        vel *= disM * self.d * self.dt
        idx2 = arange(8)

        #TODO: arggggg!!!!!
        for i in range(8):
            self.vel[i] += vel[idx2 + i * 8, :].sum(axis=0)

        #Gravity
        self.vel += self.gravVect * self.dt

        #Dumping
        self.vel -= self.vel * self.dumping * self.dt

        #velos to positions
        self.pos += self.dt * self.vel

        #Collisions and friction
        for i in range(8):
            if self.pos[i][1] < 0.0:
                self.pos[i][1] = 0.0
                self.vel[i] = self.vel[i] * [0.0, -1.0, 0.0]
        self.centerOfGrav = self.pos.sum(axis=0) / 8.0

        #Distances of new state
        idx0 = arange(8).repeat(8)
        idx1 = array(range(8) * 8)
        self.difM = self.pos[idx0, :] - self.pos[idx1, :] #vectors from all points to all other points
        self.distM = sqrt((self.difM ** 2).sum(axis=1)).reshape(64, 1) #distance matrix

    @threaded()
    def updateRenderer(self):
        self.updateDone = False
        if not self.updateLock.acquire(False): return

        # Listen for clients
        self.server.listen()
        if self.server.clients > 0:
            # If there are clients send them the new data
            self.server.send(repr([self.pos, self.centerOfGrav]))
        sleep(0.02)
        self.updateLock.release()
        self.updateDone = True

