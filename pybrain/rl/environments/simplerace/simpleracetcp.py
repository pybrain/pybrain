__author__ = 'Julian Togelius, julian@idsia.ch'

from pybrain.rl.environments import Environment
from math import sqrt
import socket
import string
from scipy import zeros

class SimpleraceEnvironment(Environment):

    firstCarScore = 0
    secondCarScore = 0
    lastStepCurrentWp = [0, 0]
    lastStepNextWp = [0, 0]

    indim = 2
    outdim = 7

    def __init__(self, host="127.0.0.1", port=6524):
        self.theSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.theSocket.connect((host, port))
        self.step = 0
        print("Connected to a simplerace server")
        self.reset()
        self.serverIsReady = False

    def getSensors(self):
        return self.sensors

    def performAction(self, action):
        # there is a nicer way of doing the following, but i'll wait with that until
        # i'm a bit more fluent in Python
        if (action[0] > 0.3):
            if(action[1]) > 0.3:
                command = 8
            elif(action[1]) < -0.3:
                command = 6
            else:
                command = 7
        elif (action[0] < -0.3):
            if(action[1]) > 0.3:
                command = 2
            elif(action[1]) < -0.3:
                command = 0
            else:
                command = 1
        else:
            if(action[1]) > 0.3:
                command = 5
            elif(action[1]) < -0.3:
                command = 3
            else:
                command = 4
        if self.waitOne:
            print('Waiting one step')
            self.waitOne = False
        elif self.serverIsReady:
            self.theSocket.send (str(command) + "\n")
        else:
            print("not sending")
        # get and process the answer
        data = ""
        while len (data) < 2:
            data = self.theSocket.recv(1000)
        #print("received", data)
        inputs = string.split(str(data), " ")
        if (inputs[0][:5] == "reset"):
            print("Should we reset the scores here?")
            self.reset ()
            self.serverIsReady = True
            self.waitOne = True
        elif (inputs[0] == "data"):
            inputs[2:20] = map(float, inputs[2:20])
            self.sensors = inputs[2:9]
            currentWp = [inputs[18], inputs[19]]
            # check that this is not the first step of an episode
            if (self.lastStepCurrentWp[0] != 0):
                # check if a way point position has changed
                if (currentWp[0] != self.lastStepCurrentWp[0]):
                    # check that we don't have a server side change of episode
                    if (currentWp[0] != self.lastStepNextWp[0]):
                        print("%.3f   %.3f   %.3f   %.3f   " % (currentWp[0], currentWp[1], self.lastStepNextWp[0], self.lastStepNextWp[1]))
                        raise Exception("Unexpected episode change")
                    else:
                        # all is fine, increase score. but for who?
                        ownPosition = [inputs[9], inputs[10]]
                        otherPosition = [inputs[14], inputs[15]]
                        if (self.euclideanDistance(ownPosition, self.lastStepCurrentWp) < self.euclideanDistance(otherPosition, self.lastStepCurrentWp)):
                            self.firstCarScore += 1
                        else:
                            self.secondCarScore += 1
            # store old way point positions
            self.lastStepCurrentWp = currentWp
            self.step += 1
        elif (len (inputs[0]) < 2):
            print("impossible!")
        else:
            print("incomprehensible and thus roundly ignored", data)

    def reset(self):
        self.step = 0
        self.firstCarScore = 0
        self.secondCarScore = 0
        self.lastStepCurrentWp = [0, 0]
        self.lastStepNextWp = [0, 0]
        self.sensors = zeros(self.outdim)
        self.waitOne = False

    def euclideanDistance(self, firstPoint, secondPoint):
        return sqrt ((firstPoint[0] - secondPoint[0]) ** 2 + (firstPoint[1] - secondPoint[1]) ** 2)
