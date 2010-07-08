__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.utilities import abstractMethod


class Environment(object):
    """ The general interface for whatever we would like to model, learn about,
        predict, or simply interact in. We can perform actions, and access
        (partial) observations.
    """

    # the number of action values the environment accepts
    indim = 0

    # the number of sensor values the environment produces
    outdim = 0

    # discrete state space
    discreteStates = False

    # discrete action space
    discreteActions = False

    # number of possible actions for discrete action space
    numActions = None

    def getSensors(self):
        """ the currently visible state of the world (the observation may be
            stochastic - repeated calls returning different values)

            :rtype: by default, this is assumed to be a numpy array of doubles
            :note: This function is abstract and has to be implemented.
        """
        abstractMethod()

    def performAction(self, action):
        """ perform an action on the world that changes it's internal state (maybe
            stochastically).
            :key action: an action that should be executed in the Environment.
            :type action: by default, this is assumed to be a numpy array of doubles
            :note: This function is abstract and has to be implemented.
        """
        abstractMethod()

    def reset(self):
        """ Most environments will implement this optional method that allows for
            reinitialization.
        """


