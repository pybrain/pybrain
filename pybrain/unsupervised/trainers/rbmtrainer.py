__author__ = ('Christian Osendorfer, osendorf@in.tum.de;'
              'Justin S Bayer, bayerj@in.tum.de'
              'SUN Yi, yi@idsia.ch')

from scipy import array, random, outer, zeros

from pybrain.structure.networks.rbm import Rbm
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
	def __init__(self, rbm):
		self.rbm = rbm
		self._invRbm = self._invertRbm()

	def _invertRbm(self):
		invRbm = Rbm(self.rbm.hiddendim, self.rbm.visibledim)
		invRbm.weights = self.rbm.weights.T
		return invRbm

	def _getData(self):
		return self._data
		
	def _setData(self, dataset):
		if dataset.indim != self.rbm.visible:
			raise ValueError("Wrong dimension for dataset")
		self._data = dataset
	
	data = property(_getData, _setData)
	
	def trainOnDataset(self, dataset, cfg = None):
		"""This function trains the RBM using the same algorithm and
		implementation presented in:
		http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html
		"""
		if cfg == None: cfg = RbmGibbsTrainerConfig()

		for rows in dataset.randomBatches('sample', cfg.batchSize):
	        olduw, olduhb, olduvb = \
	        	zeros((self.rbm.visibledim, self.rbm.hiddendim)), \
	        	zeros(self.rbm.hiddendim), zeros(self.rbm.visibledim)
	        	
			for t in xrange(cfg.maxIter):
				mm = cfg.iniMm if t < cfg.mmSwitchIter else cfg.finMm

				w, hb, vb = self.calcUpdateByRows(rows)
				
				olduw = uw = olduw * mm + \
					cfg.rWeights * (w - cfg.weightCost * self.rbm.weights)
				olduhb = uhb = olduhb * mm + cfg.rHidBias * hb
				olduvb = uvb = olduvb * mm + cfg.rVisBias * vb

				# update the parameter
				self.rbm.weights += uw
				self.rbm.biasWeights += uhb
				self._invRbm.biasWeights += uvb		
	
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
        sampled = poshp > random.rand(1, self.rbm.hiddendim)
        sampled = sampled.astype('int32')		
		recon = self._invRbm.activate(sampled)	# the re-construction of data
		
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
        	zeros((self.rbm.visibledim, self.rbm.hiddendim)), \
        	zeros(self.rbm.hiddendim), zeros(self.rbm.visibledim)
        
        for row in rows:
        	dw, dhb, dvb = self.calcUpdateRyRow(row)
        	delta_w += dw
        	delta_hb += dhb
        	delta_vb += dvb
        
        delta_w /= len(rows)
        delta_hb /= len(rows)
        delta_vb /= len(rows)

        # !!! note that this delta is only the 'theoretical' delta
        return delta_w, delta_hb, delta_vb

class RbmTrainer(Trainer):

    def __init__(self, rbm):
        self.rbm = rbm
        self._invRbm = Rbm(self.rbm.hiddendim, self.rbm.visibledim)
        self._invRbm.weights = self.rbm.weights.T
    
    def _getData(self): 
        return self._data
        
    def _setData(self, dataset):
        if not dataset.indim == self.rbm.visible:
            raise ValueError("Wrong dimension for dataset")   
        self._data = dataset
        
    data = property(_getData, _setData)
    
    def trainOnDataset(self, dataset, learningrate=.1, batchsize=50):
        for rows in dataset.randomBatches('sample', batchsize):
            uWeights, uHidBias, uVisBias = self.calcUpdateByRows(rows)
            # Normalization and learning rate factor
            a = learningrate / len(rows)
            self.rbm.weights += a * uWeights
            self._invRbm.biasWeights += a * uHidBias
            self.rbm.biasWeights += a * uVisBias
    
    def calcUpdateByRow(self, row):
        """Return a 3-tuple consiting of updates for (weightmatrix, 
        hidden bias weights, visible bias weights)."""
        # Probabilities of hidden layer in positive phase
        poshp = self.rbm.activate(row)
        # Stochastic Binary Units: Bernoulli sampling
        sampled = poshp > random.rand(1, self.rbm.hiddendim)
        sampled = sampled.astype('int32')
        # "Reconstruction" of the input
        recons = self._invRbm.activate(sampled)
        # Probabilities of hidden layer in negative phase
        neghp = self.rbm.activate(recons)
        # Gradient update for weights
        pos = outer(row, poshp)       # Fraction from the positive phase
        neg = outer(recons, neghp)    # Fraction from the first CD step
        return pos - neg, poshp - neghp, row - recons
        
    def calcUpdateByRows(self, rows):
        """Return a 3-tuple constisting of update for (weightmatrix, 
        hidden bias weights, visible bias weights)."""
        # Probabilities of hidden layer in positive phase
        poshp = array([self.rbm.activate(row) for row in rows])
        # Stochastic Binary Units: Bernoulli sampling
        sample_shape = rows.shape[0], self.rbm.hiddendim
        sampled = poshp > random.random(sample_shape)
        sampled = sampled.astype('int32')
        # "Reconstruction" of the input
        recons = array([self._invRbm.activate(sample) for sample in sampled])
        # Probabilities of hidden layer in negative phase
        neghp = array([self.rbm.activate(row) for row in recons])
        # Gradient update for weights
        pos = outer(row, poshp.T)    # Fraction from the positive phase
        neg = outer(recons, neghp.T) # Fraction from the first CD step
        # Gradient update for hidden bias 
        poshact = poshp.sum(axis=0)
        neghact = neghp.sum(axis=0)
        # Gradient update for visual bias
        posvact = rows.sum(axis=0)
        negvact = recons.sum(axis=0)

        try:
            weightupdate = pos - neg
        except: 
            print pos.shape, neg.shape, 0
            return 0, 0, 0
        hiddenbiasupdate = posvact - negvact 
        visiblebiasupdate = poshact - neghact
        return weightupdate, hiddenbiasupdate, visiblebiasupdate
