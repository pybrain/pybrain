__author__ = ('Christian Osendorfer, osendorf@in.tum.de;'
              'Justin S Bayer, bayerj@in.tum.de'
              'SUN Yi, yi@idsia.ch')

from scipy import random, outer, zeros, ones

from pybrain.datasets import SupervisedDataSet, UnsupervisedDataSet
from pybrain.supervised.trainers import Trainer
from pybrain.utilities import abstractMethod


class RbmGibbsTrainerConfig:
    def __init__(self):
        self.batchSize = 10		# how many samples in a batch

        # training rate
        self.rWeights = 0.1
        self.rHidBias = 0.1
        self.rVisBias = 0.1

        # Several configurations, I have no idea why they are here...
        self.weightCost = 0.0002

        self.iniMm = 0.5		# initial momentum
        self.finMm = 0.9		# final momentum
        self.mmSwitchIter = 5	# at which iteration we switch the momentum
        self.maxIter = 9		# how many iterations

        self.visibleDistribution = 'bernoulli'


class RbmGibbsTrainer(Trainer):
    """Class for training rbms with contrastive divergence."""

    def __init__(self, rbm, dataset, cfg=None):
        self.rbm = rbm
        self.invRbm = rbm.invert()
        self.dataset = dataset
        self.cfg = RbmGibbsTrainerConfig() if cfg is None else cfg

        if isinstance(self.dataset, SupervisedDataSet):
            self.datasetField = 'input'
        elif isinstance(self.dataset, UnsupervisedDataSet):
            self.datasetField = 'sample'

    def train(self):
        self.trainOnDataset(self.dataset)

    def trainOnDataset(self, dataset):
        """This function trains the RBM using the same algorithm and
        implementation presented in:
        http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html"""
        cfg = self.cfg
        for rows in dataset.randomBatches(self.datasetField, cfg.batchSize):
            olduw, olduhb, olduvb = \
                zeros((self.rbm.visibleDim, self.rbm.hiddenDim)), \
                zeros(self.rbm.hiddenDim), zeros(self.rbm.visibleDim)

            for t in xrange(cfg.maxIter):
                #print("*** Iteration %2d **************************************" % t)

                params = self.rbm.params
                params = params.reshape((self.rbm.visibleDim, self.rbm.hiddenDim))
                biasParams = self.rbm.biasParams

                mm = cfg.iniMm if t < cfg.mmSwitchIter else cfg.finMm

                w, hb, vb = self.calcUpdateByRows(rows)

                #print("Delta: ")
                #print("Weight: ",)
                #print(w)
                #print("Visible bias: ",)
                #print(vb)
                #print("Hidden bias: ",)
                #print(hb)
                #print("")

                olduw = uw = olduw * mm + \
                	cfg.rWeights * (w - cfg.weightCost * params)
                olduhb = uhb = olduhb * mm + cfg.rHidBias * hb
                olduvb = uvb = olduvb * mm + cfg.rVisBias * vb

                #print("Delta after momentum: ")
                #print("Weight: ",)
                #print(uw)
                #print("Visible bias: ",)
                #print(uvb)
                #print("Hidden bias: ",)
                #print(uhb)
                #print("")

                # update the parameters of the original rbm
                params += uw
                biasParams += uhb

                # Create a new inverted rbm with correct parameters
                invBiasParams = self.invRbm.biasParams
                invBiasParams += uvb
                self.invRbm = self.rbm.invert()
                self.invRbm.biasParams[:] = invBiasParams

                #print("Updated ")
                #print("Weight: ",)
                #print(self.rbm.connections[self.rbm['visible']][0].params.reshape( \)
                #    (self.rbm.indim, self.rbm.outdim))
                #print("Visible bias: ",)
                #print(self.invRbm.connections[self.invRbm['bias']][0].params)
                #print("Hidden bias: ",)
                #print(self.rbm.connections[self.rbm['bias']][0].params)
                #print("")

    def calcUpdateByRow(self, row):
        """This function trains the RBM using only one data row.
        Return a 3-tuple consiting of updates for (weightmatrix,
        hidden bias weights, visible bias weights)."""

        # a) positive phase
        poshp = self.rbm.activate(row)	# compute the posterior probability
        pos = outer(row, poshp)       	# fraction from the positive phase
        poshb = poshp
        posvb = row

        # b) the sampling & reconstruction
        sampled = self.sampler(poshp)
        recon = self.invRbm.activate(sampled)	# the re-construction of data

        # c) negative phase
        neghp = self.rbm.activate(recon)
        neg = outer(recon, neghp)
        neghb = neghp
        negvb = recon

        # compute the raw delta
        # !!! note that this delta is only the 'theoretical' delta
        return self.updater(pos, neg, poshb, neghb, posvb, negvb)

    def sampler(self, probabilities):
        abstractMethod()

    def updater(self, pos, neg, poshb, neghb, posvb, negvb):
        abstractMethod()

    def calcUpdateByRows(self, rows):
        """Return a 3-tuple constisting of update for (weightmatrix,
        hidden bias weights, visible bias weights)."""

        delta_w, delta_hb, delta_vb = \
            zeros((self.rbm.visibleDim, self.rbm.hiddenDim)), \
            zeros(self.rbm.hiddenDim), zeros(self.rbm.visibleDim)

        for row in rows:
            dw, dhb, dvb = self.calcUpdateByRow(row)
            delta_w += dw
            delta_hb += dhb
            delta_vb += dvb

        delta_w /= len(rows)
        delta_hb /= len(rows)
        delta_vb /= len(rows)

        # !!! note that this delta is only the 'theoretical' delta
        return delta_w, delta_hb, delta_vb


class RbmBernoulliTrainer(RbmGibbsTrainer):

    def sampler(self, probabilities):
        result = probabilities > random.rand(self.rbm.hiddenDim)
        return result.astype('int32')

    def updater(self, pos, neg, poshb, neghb, posvb, negvb):
        return pos - neg, poshb - neghb, posvb - negvb


class RbmGaussTrainer(RbmGibbsTrainer):

    def __init__(self, rbm, dataset, cfg=None):
        super(RbmGaussTrainer, self).__init__(rbm, dataset, cfg)
        #samples = self.dataset[self.datasetField]
        # self.visibleVariances = samples.var(axis=0)
        self.visibleVariances = ones(rbm.net.outdim)

    def sampler(self, probabilities):
        return random.normal(probabilities, self.visibleVariances)

    def updater(self, pos, neg, poshb, neghb, posvb, negvb):
        pos = pos / self.visibleVariances
        return pos - neg, poshb - neghb, posvb - negvb



