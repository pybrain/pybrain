__author__ = 'Tom Schaul, tom@idsia.ch'

"""
Adaptation of the Acrobot Environment
from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0).
"""
    
from scipy import pi, array, cos, sin
from pybrain.rl.environments.episodic import EpisodicTask


class AcrobotTask(EpisodicTask): 
    """ TODO: not currently episodic: success just reinitializes it. """   
    input_ranges = [[-pi, pi], [-pi, pi], [-4 * pi, 4 * pi], [-9 * pi, 9 * pi]]
    reward_ranges = [[-1.0, 1000.0]]

    # The current real values of the state
    maxSpeed1 = 4 * pi
    maxSpeed2 = 9 * pi
    m1 = 1.0
    m2 = 1.0
    l1 = 1.0
    l2 = 1.0
    l1Square = l1 * l1
    l2Square = l2 * l2
    lc1 = 0.5
    lc2 = 0.5
    lc1Square = lc1 * lc1
    lc2Square = lc2 * lc2
    I1 = 1.0
    I2 = 1.0
    g = 9.8
    delta_t = 0.05

    #The number of actions.
    action_list = (-1.0 , 0.0 , 1.0)
    nactions = len(action_list)
    
    # angles, velocities and a bias
    nsenses = 5

    # number of steps of the current trial
    steps = 0
    maxSteps = 999

    # number of the current episode
    episode = 0
    
    target = 1.5
    
    easy_rewards = False    
    
    resetOnSuccess = False
    
    def __init__(self):
        self.reset()
        self.cumreward = 0
                
    def getObservation(self):    
        return array(self.state + [pi])/(pi)
        
    def performAction(self, action):
        if self.done > 0:
            self.done += 1            
        else:
            self.state = self.DoAction(action, self.state)
            self.r, self.done = self.GetReward(self.state)        
            self.cumreward += self.r
            
    def reset(self):
        self.state = self.GetInitialState()
            
    def getReward(self):
        return self.r
    
    def isFinished(self):
        if self.done>=3 and self.resetOnSuccess:
            self.reset()
            return False
        else:
            return self.done>=3
    
    def GetInitialState(self):
        s = [0, 0, 0, 0]
        self.StartEpisode()
        return  s

    def StartEpisode(self):
        self.steps = 0
        self.episode = self.episode + 1
        self.done = 0

    def GetReward(self, x):
        # r: the returned reward.
        # f: true if the car reached the goal, otherwise f is false
        y_acrobot = [0, 0, 0]

        theta1 = x[0]
        theta2 = x[1]
        y_acrobot[1] = y_acrobot[0] - cos(theta1)
        y_acrobot[2] = y_acrobot[1] - cos(theta2)
        #print(y_acrobot)
        #goal
        goal = y_acrobot[0] + self.target
        if self.easy_rewards:
            r = y_acrobot[2]
        else:
            #r = -0.01
            r = 0
        f = 0

        if  y_acrobot[2] >= goal:
            if self.easy_rewards:
                r = 10 * y_acrobot[2]
            else:
                r = 1
            f = 1

        if self.steps >= self.maxSteps:
            f = 5
            #r = -1

        return r, f

    def DoAction(self, a, x):
        self.steps = self.steps + 1
        torque = self.action_list[a]

        # Parameters for simulation
        theta1, theta2, theta1_dot, theta2_dot = x

        for _ in range(4):
            d1 = self.m1 * self.lc1Square + self.m2 * (self.l1Square + self.lc2Square + 2 * self.l1 * self.lc2 * cos(theta2)) + self.I1 + self.I2
            d2 = self.m2 * (self.lc2Square + self.l1 * self.lc2 * cos(theta2)) + self.I2

            phi2 = self.m2 * self.lc2 * self.g * cos(theta1 + theta2 - pi / 2.0)
            phi1 = -self.m2 * self.l1 * self.lc2 * theta2_dot * sin(theta2) * (theta2_dot - 2 * theta1_dot) + (self.m1 * self.lc1 + self.m2 * self.l1) * self.g * cos(theta1 - (pi / 2.0)) + phi2

            accel2 = (torque + phi1 * (d2 / d1) - self.m2 * self.l1 * self.lc2 * theta1_dot * theta1_dot * sin(theta2) - phi2)
            accel2 = accel2 / (self.m2 * self.lc2Square + self.I2 - (d2 * d2 / d1))
            accel1 = -(d2 * accel2 + phi1) / d1

            theta1_dot = theta1_dot + accel1 * self.delta_t

            if theta1_dot < -self.maxSpeed1:
                theta1_dot = -self.maxSpeed1

            if theta1_dot > self.maxSpeed1:
                theta1_dot = self.maxSpeed1

            theta1 = theta1 + theta1_dot * self.delta_t
            theta2_dot = theta2_dot + accel2 * self.delta_t

            if theta2_dot < -self.maxSpeed2:
                theta2_dot = -self.maxSpeed2

            if theta2_dot > self.maxSpeed2:
                theta2_dot = self.maxSpeed2

            theta2 = theta2 + theta2_dot * self.delta_t

        # bounded angles?
        #if theta1 < -pi:
        #    theta1 = -pi
        #elif theta1 > pi:
        #    theta1 = pi
        if theta1 < -pi:
            theta1 += 2*pi
        elif theta1 > pi:
            theta1 -= 2*pi
        if theta2 < -pi:
            theta2 += 2*pi
        elif theta2 > pi:
            theta2 -= 2*pi

        xp = [theta1, theta2, theta1_dot, theta2_dot]

        return xp


class SimpleAcrobot(AcrobotTask):
    
    target = -0.5
    
class VerySimpleAcrobot(AcrobotTask):
    
    target = -1.


class SingleArmSwinger(AcrobotTask):
    """ Variant with one piece fixed."""
    
    nsenses = 3
    
    resetOnSuccess = False
    
    target = 1.95
    maxSteps = 99

    def GetInitialState(self):
        
        s = [pi, 0, 0, 0]
        self.StartEpisode()
        return  s
    
    def getObservation(self):    
        return array(self.state[2:] + [pi])/(pi)
        
    def performAction(self, action):
        AcrobotTask.performAction(self, action)
        # re-fix the upper part of the arm
        _, theta2, _, theta2_dot = self.state
        self.state = [pi, theta2, 0, theta2_dot]

        
