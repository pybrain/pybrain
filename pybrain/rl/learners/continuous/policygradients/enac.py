__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'


from policygradient import PolicyGradientLearner
from scipy import ones, dot
from scipy.linalg import pinv


class ENAC(PolicyGradientLearner):
    """ Episodic Natural Actor-Critic. See J. Peters "Natural Actor-Critic", 2005.
        Estimates natural gradient with regression of log likelihoods to rewards.
    """
            
    def _calculateGradient(self):
        # normalize rewards
        # self.ds.data['reward'] /= max(ravel(abs(self.ds.data['reward'])))
        
        # initialize variables
        R = ones((self.dataset.getNumSequences(), 1), float)
        X = ones((self.dataset.getNumSequences(), self.dataset.getDimension('loglh')+1), float)

        # collect sufficient statistics
        for n in range(self.dataset.getNumSequences()):
            _state, _action, reward, loglh = self.dataset.getSequence(n)
            X[n, :-1] = sum(loglh, 0)
            R[n, 0] = sum(reward, 0)
        
        # linear regression
        beta = dot(pinv(X), R)
        
        return beta[:-1]
        