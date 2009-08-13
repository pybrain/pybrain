__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from learning import LearningAgent


class FlatNetworkAgent(LearningAgent):
    """ a simple functional agent that contains a flat neural network to produce actions from
        observations, and does not learn. """
        
    def __init__(self, indim, outdim):        
        from pybrain.tools.shortcuts import buildNetwork
        LearningAgent.__init__(self, buildNetwork(indim, outdim))
