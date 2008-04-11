__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy.optimize import fmin

from blackboxoptimizer import BlackBoxOptimizer


class NelderMead(BlackBoxOptimizer):
    """ do the optimization using a simple wrapper for scipy's fmin """
    
    plotsymbol = '+'
    
    def optimize(self):    
        if self.tfun != None: 
            self.tfun.reset()
        return fmin(func = self.targetfun, x0 = self.x0, ftol = self.stopPrecision)