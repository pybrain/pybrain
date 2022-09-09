__author__ = 'Tom Schaul, tom@idsia.ch'

"""
Adaptation of the MountainCar Environment
from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0).
"""
    
from scipy import array, cos
from pybrain.rl.environments.episodic import EpisodicTask


class MountainCar(EpisodicTask): 
    # The current real values of the state
    cur_pos = -0.5
    cur_vel = 0.0
    cur_state = [cur_pos, cur_vel]

    #The number of actions.
    action_list = (-1.0 , 0.0 , 1.0)
    nactions = len(action_list)
    
    nsenses = 3

    # number of steps of the current trial
    steps = 0

    # number of the current episode
    episode = 0

    # Goal Position
    goalPos = 0.45
    
    maxSteps = 999
    
    resetOnSuccess = False

    def __init__(self):
        self.nactions = len(self.action_list)
        self.reset()
        self.cumreward = 0

    def reset(self):
        self.state = self.GetInitialState()
    
    def getObservation(self):    
        #print(array([self.state[0], self.state[1] * 100, 1]))
        return array([self.state[0], self.state[1] * 100, 1])
        
    def performAction(self, action):
        if self.done > 0:
            self.done += 1            
        else:
            self.state = self.DoAction(action, self.state)
            self.r, self.done = self.GetReward(self.state)
            self.cumreward += self.r
            
    def getReward(self):
        return self.r    

    def GetInitialState(self):
        self.StartEpisode()
        return [-0.5, 0.]

    def StartEpisode(self):
        self.steps = 0
        self.episode = self.episode + 1
        self.done = 0
        
    def isFinished(self):
        if self.done>=3 and self.resetOnSuccess:
            self.reset()
            return False
        else:
            return self.done>=3
    

    def GetReward(self, s):
        # MountainCarGetReward returns the reward at the current state
        # x: a vector of position and velocity of the car
        # r: the returned reward.
        # f: true if the car reached the goal, otherwise f is false

        position = s[0]
        vel = s[1]
        # bound for position; the goal is to reach position = 0.45
        bpright = self.goalPos

        r = 0
        f = 0

        if  position >= bpright:
            r = 1
            f = 1
            
        if self.steps >= self.maxSteps:
            f = 5

        return r, f

    def DoAction(self, a, s):
        #MountainCarDoAction: executes the action (a) into the mountain car
        # acti: is the force to be applied to the car
        # x: is the vector containning the position and speed of the car
        # xp: is the vector containing the new position and velocity of the car
        #print('action',a)
        #print('state',s)
        force = self.action_list[a]

        self.steps = self.steps + 1

        position = s[0]
        speed = s[1]

        # bounds for position
        bpleft = -1.4

        # bounds for speed
        bsleft = -0.07
        bsright = 0.07

        speedt1 = speed + (0.001 * force) + (-0.0025 * cos(3.0 * position))
        
        if speedt1 < bsleft:
            speedt1 = bsleft
        elif speedt1 > bsright:
            speedt1 = bsright

        post1 = position + speedt1

        if post1 <= bpleft:
            post1 = bpleft
            speedt1 = 0.0

        return [post1, speedt1]

