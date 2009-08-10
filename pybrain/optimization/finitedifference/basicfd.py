__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from scipy import ones, zeros, mean, dot, ravel
from scipy.linalg import pinv
from scipy import random
from finitediff import FDLearner
from pybrain.auxiliary import GradientDescent

class FDBasic(FDLearner):
    
    epsilon = 1.0
    gamma = 0.999
        
    def _additionalInit(self):
        self.gd = GradientDescent()

    def setModule(self, module):
        FDLearner.setModule(self, module)        
        self.gd.init(self.original)

    def perturbate(self):
        """ perturb the parameters. """
        # perturb the parameters and store the deltas in dataset
        deltas = random.uniform(-self.epsilon, self.epsilon, self.module.paramdim)
        # reduce epsilon by factor gamma
        self.epsilon *= self.gamma
        self.ds.append('deltas', deltas)
        # change the parameters in module (params is a pointer!)
        params = self.module.params
        params[:] = self.original + deltas

    def learn(self):
        """ calls the gradient calculation function and executes a step in direction
            of the gradient, scaled with a small learning rate alpha. """
        assert self.ds != None
        assert self.module != None
        
        # get the deltas from the dataset
        deltas = self.ds.getField('deltas')
        
        # initialize matrix D and vector R
        D = ones((self.ds.getNumSequences(), self.module.paramdim + 1))
        R = zeros((self.ds.getNumSequences(), 1))
        
        # calculate the gradient with pseudo inverse
        for seq in range(self.ds.getNumSequences()):
            _state, _action, reward = self.ds.getSequence(seq)
            D[seq,:-1] = deltas[seq,:]
            R[seq,:] = mean(reward)
        
        beta = dot(pinv(D), R)        
        gradient = ravel(beta[:-1])
        
        # update the weights
        self.original = self.gd(gradient)       
        self.module._setParameters(self.original)
           
        self.module.reset()
        
