""" plot the re-read results produced by femexperiments. """

__author__ = 'Tom Schaul, tom@idsia.ch'

import pylab    
from scipy import array, zeros, size, log, exp

from nesexperiments import pickleReadDict


if __name__ == '__main__':
    
    res = pickleReadDict('../temp/fem/dim15results')
    for k, val in res.items():
        allmuevals = val[2]
        if len(allmuevals):
            maxlen = max(map(len, allmuevals))
            merged = zeros(maxlen)
            for me in allmuevals:
                tmp = array(me)
                merged[:size(tmp)] += log(-tmp)
            merged /= len(allmuevals)
            merged.clip(max = log(1e10), min = log(1e-10))
            merged = exp(merged)
            x = array(range(maxlen))*50
            pylab.semilogy()
            pylab.plot(x, merged, label = k)
        
    pylab.ylabel('-fitness')
    pylab.xlabel('number of evaluations')
    pylab.legend()
    pylab.show()
    
   