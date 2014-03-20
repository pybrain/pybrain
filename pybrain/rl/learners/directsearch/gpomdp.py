__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from policygradient import PolicyGradientLearner
from scipy import zeros, mean

### NOT WORKING YET ###

class GPOMDP(PolicyGradientLearner):
    def __init__(self):
        PolicyGradientLearner.__init__(self)

    def calculateGradient(self):

        # normalize rewards
        # self.ds.data['reward'] /= max(ravel(abs(self.ds.data['reward'])))

        g = zeros((self.ds.getNumSequences(), self.ds.getDimension('loglh')), float)

        # get maximal length
        maxlen = max([self.ds.getSequenceLength(n) for n in range(self.ds.getNumSequences())])
        baselines = zeros((maxlen, self.ds.getDimension('loglh')), float)
        seqcount = zeros((maxlen, 1))

        # calculcate individual baseline for each timestep and episode
        for seq in range(self.ds.getNumSequences()):
            _, _, rewards, loglhs = self.ds.getSequence(seq)
            for t in range(len(rewards)):
                baselines[t, :] += mean(sum(loglhs[:t + 1, :], 0) ** 2 * rewards[t, :], 0) / mean(sum(loglhs[:t + 1, :], 0) ** 2, 0)
                seqcount[t, :] += 1

        baselines = baselines / seqcount
        # print(baselines)
        for seq in range(self.ds.getNumSequences()):
            _, _, rewards, loglhs = self.ds.getSequence(seq)
            for t in range(len(rewards)):
                g[seq, :] += sum(loglhs[:t + 1, :], 0) * (rewards[t, :] - baselines[t])

        gradient = mean(g, 0)
        return gradient
