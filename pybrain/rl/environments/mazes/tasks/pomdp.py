__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import ndarray

from pybrain.rl.environments import EpisodicTask
from pybrain.utilities import Named, drawIndex


class POMDPTask(EpisodicTask, Named):
    """ Partially observable episodic MDP (with discrete actions)
    Has actions that can be performed, and observations in every state.
    By default, the observation is a vector, and the actions are integers.
    """
    # number of observations
    observations = 4

    # number of possible actions
    actions = 4

    # maximal number of steps before the episode is stopped
    maxSteps = None

    # the lower bound on the reward value
    minReward = 0

    def __init__(self, **args):
        self.setArgs(**args)
        self.steps = 0

    @property
    def indim(self):
        return self.actions

    @property
    def outdim(self):
        return self.observations

    def reset(self):
        self.steps = 0
        EpisodicTask.reset(self)

    def isFinished(self):
        if self.maxSteps != None:
            return self.steps >= self.maxSteps
        return False

    def performAction(self, action):
        """ POMDP tasks, as they have discrete actions, can me used by providing either an index,
        or an array with a 1-in-n coding (which can be stochastic). """
        if type(action) == ndarray:
            action = drawIndex(action, tolerant = True)
        self.steps += 1
        EpisodicTask.performAction(self, action)