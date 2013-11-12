__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import clip, asarray

from pybrain.utilities import abstractMethod

class Task(object):
    """ A task is associating a purpose with an environment. It decides how to evaluate the
    observations, potentially returning reinforcement rewards or fitness values.
    Furthermore it is a filter for what should be visible to the agent.
    Also, it can potentially act as a filter on how actions are transmitted to the environment. """

    def __init__(self, environment):
        """ All tasks are coupled to an environment. """
        self.env = environment

        # limits for scaling of sensors and actors (None=disabled)
        self.sensor_limits = None
        self.actor_limits = None
        self.clipping = True

    def setScaling(self, sensor_limits, actor_limits):
        """ Expects scaling lists of 2-tuples - e.g. [(-3.14, 3.14), (0, 1), (-0.001, 0.001)] -
            one tuple per parameter, giving min and max for that parameter. The functions
            normalize and denormalize scale the parameters between -1 and 1 and vice versa.
            To disable this feature, use 'None'. """
        self.sensor_limits = sensor_limits
        self.actor_limits = actor_limits

    def performAction(self, action):
        """ A filtered mapping towards performAction of the underlying environment. """
        if self.actor_limits:
            action = self.denormalize(action)
        self.env.performAction(action)

    def getObservation(self):
        """ A filtered mapping to getSensors of the underlying environment. """
        sensors = self.env.getSensors()
        if self.sensor_limits:
            sensors = self.normalize(sensors)
        return sensors

    def getReward(self):
        """ Compute and return the current reward (i.e. corresponding to the last action performed) """
        return abstractMethod()

    def normalize(self, sensors):
        """ The function scales the parameters to be between -1 and 1. e.g. [(-pi, pi), (0, 1), (-0.001, 0.001)] """
        assert(len(self.sensor_limits) == len(sensors))
        result = []
        for l, s in zip(self.sensor_limits, sensors):
            if not l:
                result.append(s)
            else:
                result.append((s - l[0]) / (l[1] - l[0]) * 2 - 1.0)
        if self.clipping:
            clip(result, -1, 1)
        return asarray(result)

    def denormalize(self, actors):
        """ The function scales the parameters from -1 and 1 to the given interval (min, max) for each actor. """
        assert(len(self.actor_limits) == len(actors))
        result = []
        for l, a in zip(self.actor_limits, actors):
            if not l:
                result.append(a)
            else:
                r = (a + 1.0) / 2 * (l[1] - l[0]) + l[0]
                if self.clipping:
                    r = clip(r, l[0], l[1])
                result.append(r)

        return result

    @property
    def indim(self):
        return self.env.indim

    @property
    def outdim(self):
        return self.env.outdim


