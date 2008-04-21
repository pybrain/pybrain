__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from scipy import zeros, clip, asarray, sign     
from copy import deepcopy

class GradientDescent(object):
        
    def __init__(self):
        # learning rate
        self.alpha = 0.1
        
        # alpha decay (1.0 = disabled)
        self.alphadecay = 1.0
    
        # rprop parameters
        self.rprop = False
        self.deltamax = 5.0
        self.deltamin = 0.01
        self.deltanull = 0.1
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
        self.rprop_theta = self.lastgradient + self.deltanull      
    
    def __call__(self, gradient):            
        # check if gradient has correct dimensionality, then make array
        assert len(gradient) == len(self.values)
        gradient_arr = asarray(gradient)
        
        if self.rprop:
            rprop_theta = self.rprop_theta
            
            # update parameters 
            self.values += sign(gradient_arr) * rprop_theta

            # update rprop meta parameters
            dirSwitch = self.lastgradient * gradient_arr
            rprop_theta[dirSwitch > 0] *= self.etaplus
            idx =  dirSwitch < 0
            rprop_theta[idx] *= self.etaminus
            gradient_arr[idx] = 0

            # upper and lower bound for both matrices
            rprop_theta = rprop_theta.clip(min=self.deltamin, max=self.deltamax)

            # save current gradients to compare with in next time step
            self.lastgradient = gradient_arr.copy()
            
            self.rprop_theta = rprop_theta
        
        else:
            # update momentum vector (momentum = 0 clears it)
            self.momentumvector *= self.momentum
        
            # update parameters (including momentum)
            self.momentumvector += self.alpha * gradient_arr
            self.alpha *= self.alphadecay
        
            # update parameters 
            self.values += self.momentumvector
            
        return self.values

    descent = __call__
