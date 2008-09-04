__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from learning import LearningAgent
from history import HistoryAgent
from pybrain.structure import GaussianLayer, IdentityConnection, FeedForwardNetwork

# TODO: support for SoftMax output layers
# TODO: support for more complex networks, which have more than a single output module

class PolicyGradientAgent(LearningAgent):
    """ PolicyGradientAgent is a learning agent, that adds a GaussianLayer to
        its module and stores the log likelihoods (loglh) in the dataset.
    """
    
    def __init__(self, module, learner = None):
        assert isinstance(module, FeedForwardNetwork)
        assert len(module.outmodules) == 1
        
        LearningAgent.__init__(self, module, learner)
        
        # create gaussian layer
        self.explorationlayer = GaussianLayer(self.outdim, name='gauss')
        self.explorationlayer.setSigma([-2] * self.outdim)
        
        # add gaussian layer to top of network through identity connection
        out = self.module.outmodules.pop()
        self.module.addOutputModule(self.explorationlayer)
        self.module.addConnection(IdentityConnection(out, self.module['gauss']))
        self.module.sortModules()
        
        # tell learner the new module
        self.learner.setModule(self.module)
        
        # add the log likelihood (loglh) to the dataset and link it to the others
        self.history.addField('loglh', self.module.paramdim)
        self.history.link.append('loglh')
        self.loglh = None
    
    def enableLearning(self):
        LearningAgent.enableLearning(self)
        self.explorationlayer.enabled = True
    
    def disableLearning(self):
        LearningAgent.disableLearning(self)
        self.explorationlayer.enabled = False
        
    def setSigma(self, sigma):
        assert len(sigma) == self.explorationlayer.paramdim
        # change the parameters of the exploration layer (owner is self.module)
        self.explorationlayer._setParameters(sigma, self.module)
    
    def getSigma(self):
        return self.explorationlayer.params
               
    def setParameters(self, params):
        self.module._setParameters(params)
        # update parameters for learner
        self.learner.setModule(self.module)
    
    def getAction(self):
        """ calls the LearningAgent getAction method. Additionally, executes a backward pass in the module
            and stores all the derivatives in the dataset. """
        HistoryAgent.getAction(self)
        
        self.lastaction = self.module.activate(self.lastobs).copy()
        self.module.backward()
        self.loglh = self.module.derivs.copy()
        
        d = self.module.derivs
        d *= 0
        self.module.reset()
        return self.lastaction
        
    def giveReward(self, r):
        """ stores observation, action, reward and the log likelihood
            in the history dataset.
            @param r: reward for this timestep 
            @note: this function overwrites HistoryAgent.giveReward(self, r)
        """ 
        assert self.lastobs != None
        assert self.lastaction != None

        # store state, action, r, loglh in dataset
        if self.remember:
            self.history.appendLinked(self.lastobs, self.lastaction, r, self.loglh)

        self.lastobs = None
        self.lastaction = None
