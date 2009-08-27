__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from policygradient import PolicyGradientLearner
from scipy import mean


class Reinforce(PolicyGradientLearner):
    """ Reinforce is a gradient estimator technique by Williams (see
        "Simple Statistical Gradient-Following Algorithms for
        Connectionist Reinforcement Learning"). It uses optimal
        baselines and calculates the gradient with the log likelihoods
        of the taken actions. """ 
         
    def __init__(self):
        PolicyGradientLearner.__init__(self)

    def calculateGradient(self):
        # normalize rewards
        # self.ds.data['reward'] /= max(ravel(abs(self.ds.data['reward'])))
        
        # initialize variables
        returns = self.ds.getSumOverSequences('reward')
        loglhs = self.ds.getSumOverSequences('loglh')
                
        # only take better half of returns/loglhs
        # loglhs = loglhs[where(returns > mean(returns))[0],:]
        # returns = returns[where(returns > mean(returns))[0], :]
        
        baselines = mean(loglhs**2 * returns, 0) / mean(loglhs**2, 0)
        gradient = mean(loglhs * (returns-baselines), 0)
        
        return gradient
                