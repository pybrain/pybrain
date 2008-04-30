""" plot the re-read results produced by femexperiments. """

__author__ = 'Tom Schaul, tom@idsia.ch'

import pylab    
from scipy import array, zeros, size

from nesexperiments import pickleReadDict


if __name__ == '__main__':
    i = 0
    res = pickleReadDict('../temp/fem/dim15results-b')
    plotsymbols = ['-']#, ':', '-.', '-.', '-', ':']
    for k, val in sorted(res.items()):
        allmuevals = filter(lambda x: max(x) > -1e-10, val[2])
        print k, len(val[2]), len(allmuevals)
        if len(allmuevals):
            maxlen = max(map(len, allmuevals))
            merged = zeros(maxlen)
            avgover = zeros(maxlen)
            for me in allmuevals:
                tmp = array(me)
                merged[:size(tmp)] -= tmp
                avgover +=1 #[:size(tmp)] += 1
            merged /= avgover
            merged = merged.clip(min = 1e-10, max = 1e20)
            x = array(range(maxlen))*25
            pylab.semilogy()
            pylab.plot(x, merged, plotsymbols[i], label = k[:-10])
        i = (i+1) % len(plotsymbols)
    pylab.ylabel('fitness')
    pylab.xlabel('number of evaluations')
    pylab.legend()
    pylab.show()
    
   