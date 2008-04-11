__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from agent import Agent
from pybrain.datasets import ReinforcementDataSet

class HistoryAgent(Agent):
    """ This agent stores actions, states, and rewards encountered during interaction with an environment
        in a ReinforcementDataSet (which is a slight variation of SequentialDataSet). The stored history can 
        be used for learning and is erased by resetting the agent. It also makes sure that integrateObservation,
        getAction and giveReward are called in exactly that order. """
        
    def __init__(self, indim, outdim):        
        # store input and output dimension
        self.indim = indim
        self.outdim = outdim
                
        # create history dataset
        self.history = ReinforcementDataSet(indim, outdim)

        # initialize temporary variables
        self.lastobs = None
        self.lastaction = None
        
    def integrateObservation(self, obs):
        """ stores the observation received in a temporary variable until action is called and
            reward is given (STEP 1) """
        assert self.lastobs == None
        assert self.lastaction == None
        
        self.lastobs = obs
        
    def getAction(self):
        """ stores the action in a temporary variable until reward is given (STEP 2) """
        assert self.lastobs != None 
        assert self.lastaction == None
        # implement getAction in subclass and set self.lastaction
        
    def giveReward(self, r):
        """ stores observation, action and reward in the history dataset (STEP 3) """
        # step 3: assume that state and action have been set
        assert self.lastobs != None
        assert self.lastaction != None

        # store state, action and reward in dataset
        self.history.addSample(self.lastobs, self.lastaction, r)

        self.lastobs = None
        self.lastaction = None
            
    def reset(self):
        """ clears the history of the agent. """
        self.history.clear()
