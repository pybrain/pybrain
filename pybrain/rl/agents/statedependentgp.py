__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from scipy import ravel, diag, where, random
from learning import LearningAgent
from policygradient import PolicyGradientAgent
from pybrain.structure import StateDependentLayer, IdentityConnection
from pybrain import buildNetwork
from pylab import ion #@UnresolvedImport
from pybrain.auxiliary import GaussianProcess

class StateDependentAgent(PolicyGradientAgent):
    """ StateDependentAgent is a learning agent, that adds a GaussianLayer to its module and stores its
        deterministic inputs (mu) in the dataset.
    """
    
    def __init__(self, module, learner = None):
        ion()
        LearningAgent.__init__(self, module, learner)
        
        # exploration module (linear flat network)
        self.explorationmodule = buildNetwork(self.indim, self.outdim, bias=False)
        
        # state dependent exploration layer
        self.explorationlayer = StateDependentLayer(self.outdim, self.explorationmodule, 'explore')
                
        # add exploration layer to top of network through identity connection
        out = self.module.outmodules.pop()
        self.module.addOutputModule(self.explorationlayer)
        self.module.addConnection(IdentityConnection(out, self.module['explore'], self.module))
        self.module.sortModules()
        
        # tell learner the new module
        self.learner.setModule(self.module)
        
        # add the log likelihood (loglh) to the dataset and link it to the others
        self.history.addField('loglh', self.module.paramdim)
        self.history.link.append('loglh')
        self.loglh = None
        
        # if this flag is set to True, random weights are drawn after each reward,
        # effectively acting like the vanilla policy gradient alg.
        self.actaspg = False

        # gaussian process
        self.gp = GaussianProcess(1, -5, 5, 0.1)
        self.gp.autonoise = True
        self.gp.mean = -3

    def newEpisode(self):
        params = ravel(self.explorationlayer.module.params)
        target = ravel(sum(self.history.getSequence(self.history.getNumSequences()-1)[2]) / 500)
        if target != 0.0:
            self.gp.addSample(params, target)
            # self.gp.noise += 0.01
            self.gp.plotCurves()                
        LearningAgent.newEpisode(self)
        
        indices = where(diag(self.gp.pred_cov) == diag(self.gp.pred_cov).max())[0]
        new_param = self.gp.testx[indices[random.randint(len(indices))]]
        self.explorationlayer.module._setParameters([new_param])
    

    def getAction(self):
        self.explorationlayer.setState(self.lastobs)
        action = PolicyGradientAgent.getAction(self)
        return action
        
    def giveReward(self, r):
        PolicyGradientAgent.giveReward(self, r)
        if self.actaspg:
            self.explorationlayer.drawRandomWeights()