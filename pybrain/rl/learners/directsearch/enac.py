__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'


from pybrain.rl.learners.directsearch.policygradient import PolicyGradientLearner
from scipy import ones, dot, ravel
from scipy.linalg import pinv


class ENAC(PolicyGradientLearner):
    """ Episodic Natural Actor-Critic. See J. Peters "Natural Actor-Critic", 2005.
        Estimates natural gradient with regression of log likelihoods to rewards.
    """

    def calculateGradient(self):
        # normalize rewards
        # self.dataset.data['reward'] /= max(ravel(abs(self.dataset.data['reward'])))

        # initialize variables
        R = ones((self.dataset.getNumSequences(), 1), float)
        X = ones((self.dataset.getNumSequences(), self.loglh.getDimension('loglh') + 1), float)

        # collect sufficient statistics
        print(self.dataset.getNumSequences())
        for n in range(self.dataset.getNumSequences()):
            _state, _action, reward = self.dataset.getSequence(n)
            seqidx = ravel(self.dataset['sequence_index'])
            if n == self.dataset.getNumSequences() - 1:
                # last sequence until end of dataset
                loglh = self.loglh['loglh'][seqidx[n]:, :]
            else:
                loglh = self.loglh['loglh'][seqidx[n]:seqidx[n + 1], :]

            X[n, :-1] = sum(loglh, 0)
            R[n, 0] = sum(reward, 0)

        # linear regression
        beta = dot(pinv(X), R)
        return beta[:-1]

