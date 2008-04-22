__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from learning import LearningAgent
from pybrain import buildNetwork


class FlatNetworkAgent(LearningAgent):
    """ a simple functional agent that contains a flat neural network to produce actions from
        observations, and does not learn. """
        
    def __init__(self, indim, outdim):        
        LearningAgent.__init__(self, buildNetwork(indim, outdim))
