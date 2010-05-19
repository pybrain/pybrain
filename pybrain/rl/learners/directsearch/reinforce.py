__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.rl.learners.directsearch.policygradient import PolicyGradientLearner
from scipy import mean, ravel, array


class Reinforce(PolicyGradientLearner):
    """ Reinforce is a gradient estimator technique by Williams (see
        "Simple Statistical Gradient-Following Algorithms for
        Connectionist Reinforcement Learning"). It uses optimal
        baselines and calculates the gradient with the log likelihoods
        of the taken actions. """

    def calculateGradient(self):
        # normalize rewards
        # self.ds.data['reward'] /= max(ravel(abs(self.ds.data['reward'])))

        # initialize variables
        returns = self.dataset.getSumOverSequences('reward')
        seqidx = ravel(self.dataset['sequence_index'])

        # sum of sequences up to n-1
        loglhs = [sum(self.loglh['loglh'][seqidx[n]:seqidx[n + 1], :]) for n in range(self.dataset.getNumSequences() - 1)]
        # append sum of last sequence as well
        loglhs.append(sum(self.loglh['loglh'][seqidx[-1]:, :]))
        loglhs = array(loglhs)

        baselines = mean(loglhs ** 2 * returns, 0) / mean(loglhs ** 2, 0)
        # TODO: why gradient negative?
        gradient = -mean(loglhs * (returns - baselines), 0)

        return gradient

