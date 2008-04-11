from pybrain.rl.environments.cartpole.nonmarkovpole import NonMarkovPoleEnvironment
__author__ = 'Thomas Rueckstiess and Tom Schaul'

from pybrain.rl.tasks import EpisodicTask
from cartpole import CartPoleEnvironment
from scipy import pi

class BalanceTask(EpisodicTask):

    """ The task of balancing some pole(s) on a cart """
    def __init__(self, env = None, maxsteps = 1000):
        """
        @param env: (optional) an instance of a CartPoleEnvironment (or a subclass thereof)
        @param maxsteps: maximal number of steps (default: 1000) 
        """
        if env == None:
            env = CartPoleEnvironment()
        EpisodicTask.__init__(self, env) 
        self.N = maxsteps
        self.t = 0
        
        # scale position and angle, don't scale velocities (unknown maximum)
        self.sensor_limits = [(-3, 3)]#, None, (-pi, pi), None]
        for i in range(1,self.getOutDim()):
            if isinstance(self.env, NonMarkovPoleEnvironment) and i%2 == 0:
                self.sensor_limits.append(None)
            else:
                self.sensor_limits.append((-pi, pi))
        
        # actor between -10 and 10 Newton
        self.actor_limits = [(-10, 10)]
        
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



class EasyBalanceTask(BalanceTask):
    """ this task is a bit easier to learn because it gives gradual feedback
        about the distance to the centre. """
    def getReward(self):
        angles = map(abs, self.env.getPoleAngles())
        s = abs(self.env.getCartPosition())
        reward = 0
        if min(angles) < 0.05 and abs(s) < 0.05:
            reward = 0
        elif max(angles) > 0.7 or abs(s) > 2.4:
            reward = -2 * (self.N - self.t)
        else: 
            reward = -abs(s)/2
        return reward       