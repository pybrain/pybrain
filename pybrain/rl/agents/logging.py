__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.rl.agents.agent import Agent
from pybrain.datasets import ReinforcementDataSet


class LoggingAgent(Agent):
    """ This agent stores actions, states, and rewards encountered during
        interaction with an environment in a ReinforcementDataSet (which is
        a variation of SequentialDataSet).
        The stored history can be used for learning and is erased by resetting
        the agent. It also makes sure that integrateObservation, getAction and
        giveReward are called in exactly that order.
    """

    logging = True

    lastobs = None
    lastaction = None
    lastreward = None


    def __init__(self, indim, outdim, **kwargs):
        self.setArgs(**kwargs)
        
        # store input and output dimension
        self.indim = indim
        self.outdim = outdim

        # create the history dataset
        self.history = ReinforcementDataSet(indim, outdim)


    def integrateObservation(self, obs):
        """Step 1: store the observation received in a temporary variable until action is called and
        reward is given. """
        self.lastobs = obs
        self.lastaction = None
        self.lastreward = None


    def getAction(self):
        """Step 2: store the action in a temporary variable until reward is given. """
        assert self.lastobs != None
        assert self.lastaction == None
        assert self.lastreward == None

        # implement getAction in subclass and set self.lastaction


    def giveReward(self, r):
        """Step 3: store observation, action and reward in the history dataset. """
        # step 3: assume that state and action have been set
        assert self.lastobs != None
        assert self.lastaction != None
        assert self.lastreward == None

        self.lastreward = r

        # store state, action and reward in dataset if logging is enabled
        if self.logging:
            self.history.addSample(self.lastobs, self.lastaction, self.lastreward)


    def newEpisode(self):
        """ Indicate the beginning of a new episode in the training cycle. """
        if self.logging:
            self.history.newSequence()


    def reset(self):
        """ Clear the history of the agent. """
        self.lastobs = None
        self.lastaction = None
        self.lastreward = None

        self.history.clear()
