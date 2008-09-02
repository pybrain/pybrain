# $Id$
__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from scipy import zeros, asarray, sign     
from copy import deepcopy

class GradientDescent(object):
        
    def __init__(self):
        """ initialise algorithms with standard parameters (typical values given in parentheses)"""
        #-------<common parameters>--------
        #-------</common parameters>-------
        
        #-------<BackProp>-----------------
        # learning rate (0.1-0.001, down to 1e-7 for RNNs)
        self.alpha = 0.1
        
        # alpha decay (0.999; 1.0 = disabled)
        self.alphadecay = 1.0
    
        # momentum parameters (0.1 or 0.9)
        self.momentum = 0.0
        self.momentumvector = None
        #-------</BackProp>----------------

        #-------<RProp>--------------------
        self.rprop = False
        # maximum step width (1 - 20)
        self.deltamax = 5.0
        # minimum step width (0.01 - 1e-6)
        self.deltamin = 0.01
        # the remaining parameters do not normally need to be changed
        self.deltanull = 0.1
        self.etaplus = 1.2
        self.etaminus = 0.5
        self.lastgradient = None
        #-------</RProp>-------------------
        
    
        
    def init(self, values):
        """ call this to initialize data structures *after* algorithm to use
        has been selected
        @param values: the list (or array) of parameters to perform gradient descent on
                       (will be copied, original not modified)
        """
        self.values = deepcopy(values)
        if self.rprop:
            self.lastgradient = zeros(len(values))
            self.rprop_theta = self.lastgradient + self.deltanull      
            self.momentumvector = None
        else:
            self.lastgradient = None
            self.momentumvector = zeros(len(values))
            
    
    def __call__(self, gradient):            
        """ calculates parameter change based on given gradient and returns updated parameters """
        # check if gradient has correct dimensionality, then make array """
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
