__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from scipy import zeros, clip, asarray, sign     
from copy import deepcopy

class GradientDescent(object):
        
    def __init__(self):
        # learning rate
        self.alpha = 0.1
    
        # rprop parameters
        self.rprop = False
        self.deltamax = 5.0
        self.deltamin = 0.01
        self.etaplus = 1.2
        self.etaminus = 0.5
        self.lastgradient = None
        
        # momentum parameters
        self.momentum = 0.0
        self.momentumvector = None

    def init(self, values):
        self.values = deepcopy(values)
        self.momentumvector = zeros(len(values))
        self.lastgradient = zeros(len(values))
        self.rprop_theta = self.lastgradient + 0.1        
    
    def __call__(self, gradient):            
        # check if gradient has correct dimensionality, then make array
        assert len(gradient) == len(self.values)
        gradient_arr = asarray(gradient)
        
        # update momentum vector (momentum = 0 clears it)
        self.momentumvector *= self.momentum
        
        if self.rprop:
            # update parameters (including momentum) 
            self.momentumvector += sign(gradient_arr) * self.rprop_theta

            # update rprop meta parameter
            self.rprop_theta[(self.lastgradient * gradient_arr) > 0] *= self.etaplus
            self.rprop_theta[(self.lastgradient * gradient_arr) < 0] *= self.etaminus

            # upper and lower bound for both matrices
            self.rprop_theta = self.rprop_theta.clip(min=self.deltamin, max=self.deltamax)

            # save current gradients to compare with in next time step
            self.lastgradient = gradient_arr.copy()
        
        else:
            # update parameters (including momentum)
            self.momentumvector += self.alpha * gradient_arr
        
        # return the new values
        self.values += self.momentumvector
        return self.values

    descent = __call__
