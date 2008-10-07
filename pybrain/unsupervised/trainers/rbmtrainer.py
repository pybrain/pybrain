__author__ = ('Christian Osendorfer, osendorf@in.tum.de;'
              'Justin S Bayer, bayerj@in.tum.de'
              'SUN Yi, yi@idsia.ch')

from scipy import array, random, outer, zeros, dot

from pybrain.structure.networks.rbm import invRbm
from pybrain.supervised.trainers import Trainer


class RbmGibbsTrainerConfig:
    def __init__(self):
        self.batchSize = 50		# how many samples in a batch

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


class RbmGibbsTrainer(Trainer):
    def __init__(self, rbm, dataset, cfg=None):
        self.rbm = rbm
        self.invrbm = invRbm(rbm)
        self.dataset = dataset
        self.cfg = RbmGibbsTrainerConfig() if cfg is None else cfg

    def train(self):
        self.trainOnDataset(self.dataset)

    def trainOnDataset(self, dataset):
        """This function trains the RBM using the same algorithm and
        implementation presented in:
        http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html"""
        cfg = self.cfg
        for rows in dataset.randomBatches('input', cfg.batchSize):
            olduw, olduhb, olduvb = \
                zeros((self.rbm.indim, self.rbm.outdim)), \
                zeros(self.rbm.outdim), zeros(self.rbm.indim)

            for t in xrange(cfg.maxIter):
                weights = self.rbm.connections[self.rbm['visible']][0].params
                biasWeights = self.rbm.connections[self.rbm['bias']][0].params
                mm = cfg.iniMm if t < cfg.mmSwitchIter else cfg.finMm

                w, hb, vb = self.calcUpdateByRows(rows)

                olduw = uw = olduw * mm + \
                	cfg.rWeights * (w - cfg.weightCost * weights)
                olduhb = uhb = olduhb * mm + cfg.rHidBias * hb
                olduvb = uvb = olduvb * mm + cfg.rVisBias * vb

                # update the parameters of the original rbm
                print uw.shape
                weights += uw.flatten()
                biasWeights += uhb.flatten()
                # Create a new inverted rbm with correct parameters
                invBiasWeights = self._invRbm.connections[self.rbm['bias']][0].params
                invBiasWeights += uvb
                self._invRbm = invRbm(self.rbm)
                self._invRbm.connections[self.rbm['bias']][0].params = invBiasWeights

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
        sampled = poshp > random.rand(1, self.rbm.outdim)
        sampled = sampled.astype('int32')
        recon = self.invrbm.activate(sampled)	# the re-construction of data

        # c) negative phase
        neghp = self.rbm.activate(recon)
        neg = outer(recon, neghp)
        neghb = neghp
        negvb = recon

        # compute the raw delta
        # !!! note that this delta is only the 'theoretical' delta
        return pos - neg, poshb - neghb, posvb - negvb
    
    def calcUpdateByRows(self, rows):
        """Return a 3-tuple constisting of update for (weightmatrix,
        hidden bias weights, visible bias weights)."""
        
        delta_w, delta_hb, delta_vb = \
            zeros((self.rbm.indim, self.rbm.outdim)), \
            zeros(self.rbm.outdim), zeros(self.rbm.indim)
        
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