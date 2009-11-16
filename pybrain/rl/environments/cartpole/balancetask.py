__author__ = 'Thomas Rueckstiess and Tom Schaul'

from scipy import pi, dot, array

from pybrain.rl.environments.cartpole.nonmarkovpole import NonMarkovPoleEnvironment
from pybrain.rl.environments import EpisodicTask
from cartpole import CartPoleEnvironment


class BalanceTask(EpisodicTask):
    """ The task of balancing some pole(s) on a cart """   
    def __init__(self, env=None, maxsteps=1000, desiredValue = 0):
        """
        :key env: (optional) an instance of a CartPoleEnvironment (or a subclass thereof)
        :key maxsteps: maximal number of steps (default: 1000) 
        """
        self.desiredValue = desiredValue
        if env == None:
            env = CartPoleEnvironment()
        EpisodicTask.__init__(self, env) 
        self.N = maxsteps
        self.t = 0
        
        # scale position and angle, don't scale velocities (unknown maximum)
        self.sensor_limits = [(-3, 3)]#, None, (-pi, pi), None]
        for i in range(1, self.outdim):
            if isinstance(self.env, NonMarkovPoleEnvironment) and i % 2 == 0:
                self.sensor_limits.append(None)
            else:
                self.sensor_limits.append((-pi, pi))
        
        self.sensor_limits = [None] * 4
        # actor between -10 and 10 Newton
        self.actor_limits = [(-50, 50)]
        
    def reset(self):
        EpisodicTask.reset(self)
        self.t = 0

    def performAction(self, action):
        self.t += 1
        EpisodicTask.performAction(self, action)
                       
    def isFinished(self):
        if max(map(abs, self.env.getPoleAngles())) > 0.7:
            # pole has fallen
            return True
        elif abs(self.env.getCartPosition()) > 2.4:
            # cart is out of it's border conditions
            return True
        elif self.t >= self.N:
            # maximal timesteps
            return True
        return False
        
    def getReward(self):
        angles = map(abs, self.env.getPoleAngles())
        s = abs(self.env.getCartPosition())
        reward = 0
        if min(angles) < 0.05 and abs(s) < 0.05:
            reward = 0
        elif max(angles) > 0.7 or abs(s) > 2.4:
            reward = -2 * (self.N - self.t)
        else: 
            reward = -1
        return reward
        
    def setMaxLength(self, n):
        self.N = n    


class JustBalanceTask(BalanceTask):
    """ this task does not require the cart to be moved to the middle. """
    def getReward(self):
        angles = map(abs, self.env.getPoleAngles())
        s = abs(self.env.getCartPosition())
        if min(angles) < 0.05:
            reward = 0
        elif max(angles) > 0.7 or abs(s) > 2.4:
            reward = -2 * (self.N - self.t)
        else: 
            reward = -1
        return reward
    
        
class EasyBalanceTask(BalanceTask):
    """ this task is a bit easier to learn because it gives gradual feedback
        about the distance to the centre. """
    def getReward(self):
        angles = map(abs, self.env.getPoleAngles())
        s = abs(self.env.getCartPosition())
        if min(angles) < 0.05 and abs(s) < 0.05:
            reward = 0
        elif max(angles) > 0.7 or abs(s) > 2.4:
            reward = -2 * (self.N - self.t)
        else: 
            reward = -abs(s) / 2
        return reward   
    
    

class DiscreteBalanceTask(BalanceTask):
    """ here there are 3 discrete actions, left, right, nothing. """
    
    def __init__(self, env=None, maxsteps=1000):
        """
        :key env: (optional) an instance of a CartPoleEnvironment (or a subclass thereof)
        :key maxsteps: maximal number of steps (default: 1000) 
        """
        if env == None:
            env = CartPoleEnvironment()
        EpisodicTask.__init__(self, env) 
        self.N = maxsteps
        self.t = 0
        
        # no scaling of sensors
        self.sensor_limits = [None] * 2
        
        # scale actor
        self.actor_limits = [(-50, 50)]

        
    def getObservation(self):
        """ a filtered mapping to getSample of the underlying environment. """
        sensors = self.env.getSensors()[:2]   
        if self.sensor_limits:
            sensors = self.normalize(sensors)
        return sensors
        
    def performAction(self, action):
        action = action - 1.
        BalanceTask.performAction(self, action)
    
    def getReward(self):
        angles = map(abs, self.env.getPoleAngles())
        s = abs(self.env.getCartPosition())
        if min(angles) < 0.05:
            reward = 1.0
        elif max(angles) > 0.7 or abs(s) > 2.4:
            reward = -1. * (self.N - self.t)
        else: 
            reward = 0.0
        return reward




class DiscreteNoHelpTask(DiscreteBalanceTask):
    def getReward(self):
        angles = map(abs, self.env.getPoleAngles())
        s = abs(self.env.getCartPosition())
        if max(angles) > 0.7 or abs(s) > 2.4:
            reward = -1. * (self.N - self.t)
        else: 
            reward = 0.0
        return reward   
    



class LinearizedBalanceTask(BalanceTask):
    """ Here we follow the setup in
    Peters J, Vijayakumar S, Schaal S (2003) Reinforcement learning for humanoid robotics.
    TODO: This stuff is not yet compatible to any other cartpole environment. """

    Q = array([12., 0.25, 1.25, 1.0])
    
    def getReward(self):
        return dot(self.env.sensors ** 2, self.Q) + self.env.action[0] ** 2 * 0.01
    
    def isFinished(self):        
        if abs(self.env.getPoleAngles()[0]) > 0.5235988:  # pi/6
            # pole has fallen
            return True
        elif abs(self.env.getCartPosition()) > 1.5:
            # cart is out of it's border conditions
            return True
        elif self.t >= self.N:
            # maximal timesteps
            return True
        return False
    

