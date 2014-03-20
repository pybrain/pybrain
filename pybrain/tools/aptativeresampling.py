__author__ = "Tom Schaul, tom@idsia.ch"


from scipy import median


class AdaptiveResampler(object):
    """ A simplified version of the uncertainty handling method described in
    Hansen, Niederberger, Guzzella and Koumoutsakos, 2009."""
            
    def __init__(self, f, batchsize, update_factor=1.5, threshold=0.2, max_resampling=None):
        self.f = f
        self.batchsize = batchsize
        self.update_factor = update_factor
        self.threshold = threshold
        self.max_resampling = max_resampling
        
        self.recents = [None]*self.batchsize
        self.resample_over = 1
        self.num_evals = 0
                
    def __call__(self, x):
        res = median([self.f(x) for _ in range(int(self.resample_over))])
        if self.num_evals%self.batchsize == 0 and self.num_evals > 0:
            alt_res = median([self.f(x) for _ in range(int(self.resample_over))])
            self._adaptResampling(res, alt_res)
            res = 0.5*res+0.5*alt_res
        self.recents[self.num_evals%self.batchsize] = res
        self.num_evals += 1
        return res
    
    def _adaptResampling(self, res, alt):
        #compute rank change
        rc = sum([(x-res) * (x-alt) < 0 for x in self.recents[1:]])
        if rc >= self.threshold*(self.batchsize-1):
            self.resample_over *= self.update_factor 
            if self.max_resampling is not None and self.resample_over > self.max_resampling:
                self.resample_over = self.max_resampling
        elif rc == 0:
            self.resample_over = max(self.resample_over/self.update_factor, 1)
        

    
def testnes():
    from pybrain.optimization.distributionbased.xnes import XNES
    from scipy import ones
    from random import gauss    
    import pylab  
    noise = 0.1
    x0 = ones(5)
    fun = lambda x: -sum(x**2) - gauss(0,noise)
    
    fun2 = AdaptiveResampler(fun, 10)
    l = XNES(fun, x0, maxEvaluations=1100, storeAllEvaluations=True)
    res = l.learn()
    print(sum(res[0]**2) )
    pylab.plot(map(abs, l._allEvaluations))
    
    l2 = XNES(fun2, x0, maxEvaluations=1100, storeAllEvaluations=True)
    res = l2.learn()
    print(sum(res[0]**2) )
    print(fun2.resample_over)
    pylab.plot(map(abs,l2._allEvaluations))
    pylab.semilogy()
    pylab.show()
    
if __name__ == "__main__":
    testnes()    
    