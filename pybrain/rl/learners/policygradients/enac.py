__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'


from policygradient import PolicyGradientLearner
from scipy import ones, dot
from scipy.linalg import pinv


class ENAC(PolicyGradientLearner):
    """ Episodic Natural Actor-Critic"""
    
    def __init__(self):
        PolicyGradientLearner.__init__(self)
        
    def calculateGradient(self):
        # normalize rewards
        # self.ds.data['reward'] /= max(ravel(abs(self.ds.data['reward'])))
        
        # initialize variables
        R = ones((self.ds.getNumSequences(), 1), float)
        X = ones((self.ds.getNumSequences(), self.ds.getDimension('loglh')+1), float)

        # collect sufficient statistics
        for n in range(self.ds.getNumSequences()):
            state, action, reward, loglh = self.ds.getSequence(n)
            X[n, :-1] = sum(loglh, 0)
            R[n, 0] = sum(reward, 0)
        
        # linear regression
        beta = dot(pinv(X), R)
        
        return beta[:-1]
        