__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from scipy import ravel, diag, where, random

from pybrain.auxiliary import GaussianProcess
from statedependent import StateDependentAgent
from learning import LearningAgent
# from pylab import ion, show, draw, gcf, hold
# from matplotlib import axes3d as a3

class StateDependentGPAgent(StateDependentAgent):
    """ StateDependentAgent is a learning agent, that adds a GaussianLayer to its module and stores its
        deterministic inputs (mu) in the dataset. It also trains a GaussianProcess with the drawn
        exploration weights and the reward as target for choosing the next exploration sample.
    """
    
    def __init__(self, module, learner = None):
        StateDependentAgent.__init__(self, module, learner)

        # gaussian process
        self.gp = GaussianProcess(self.explorationlayer.module.paramdim, -2, 2, 1)
        self.gp.mean = -1.5
        self.gp.hyper = (2.0, 2.0, 0.1)
        #ion()
        
    def newEpisode(self):
        if self.learning:
            params = ravel(self.explorationlayer.module.params)
            target = ravel(sum(self.history.getSequence(self.history.getNumSequences()-1)[2]) / 500)
        
            if target != 0.0:
                self.gp.addSample(params, target)
                if len(self.gp.trainx) > 20:
                    self.gp.trainx = self.gp.trainx[-20:, :]
                    self.gp.trainy = self.gp.trainy[-20:]
                    self.gp.noise = self.gp.noise[-20:]
                    
                self.gp._calculate()
                # ax = self.gp.plotCurves()
                        
                # get new parameters where mean was highest
                max_cov = diag(self.gp.pred_cov).max()
                indices = where(diag(self.gp.pred_cov) == max_cov)[0]
                pick = indices[random.randint(len(indices))]
                new_param = self.gp.testx[pick]
            
                # check if that one exists already in gp training set
                if len(where(self.gp.trainx == new_param)[0]) > 0:
                   # add some normal noise to it
                   new_param += random.normal(0, 1, len(new_param))

                self.explorationlayer.module._setParameters(new_param)

                # plot new sample coordinate
                # ax.plot3D([new_param[0]], [new_param[1]], [self.gp.pred_mean[pick]], 'bo')
                # draw()
            
            else:
                self.explorationlayer.drawRandomWeights()
        
        # don't call StateDependentAgent.newEpisode() because it randomizes the params
        LearningAgent.newEpisode(self)