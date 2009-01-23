""" a type of plots used so frequently that I think they merit their own utility """

__author__ = 'Tom Schaul, tom@idsia.ch'

import pylab
from pylab import xlabel, ylabel, legend, plot, semilogy
from scipy import array, zeros


plotsymbols = ['-', ':', '-.']
psymbol = '-'

    
def plotFitnessProgession(fitdict, batchsize = 1, show = True, semilog = True, 
                          targetcutoff = 1e-10, minimize = True, 
                          onlysuccessful = True,
                          title = None, verbose = True,
                          varyplotsymbols = False):
    """ Plot multiple fitness curves on a single figure, with the following customizations:
    
        @param fitdict: a dictionnary mapping a name to a list of fitness-arrays 
        @param batchsize: the number of evaluations between two points in fitness-arrays 
                          specific batchsizes can also be given given in fitdict
        @param targetcutoff: this gives the cutoff point at the best fitness
        @param onlysuccessful: ignore the runs that did not hit the target
        @param title: specify a title.
        @param varyplotsymbols: used different line types for each curve.
        """
        
    def isSuccessful(l):
        """ criterion for successful run """
        if targetcutoff == None:
            return True
        elif minimize:
            return min(l) <= targetcutoff
        else:
            return max(l) >= targetcutoff
        
    def paddedClipped(l, maxLen):
        assert len(l) <= maxLen
        res = zeros(maxLen)
        if targetcutoff == None:
            res[:len(l)] += l
        elif minimize:
            res[:len(l)] += l.clip(min = targetcutoff, max = 1e100)            
        else:
            res[:len(l)] += l.clip(max = targetcutoff, min = -1e100)            
        return res
    
    def relevantPart(l):
        """ the part of the vector that's above the cutoff. """
        if targetcutoff != None:
            for i, val in enumerate(l):
                if minimize and val <= targetcutoff:
                    return l[:i+1]
                elif not minimize and val >= targetcutoff:
                    return l[:i+1]
        return l
            
        
        
    i = 0
    for name, flist in sorted(fitdict.items()):
        if isinstance(flist, tuple):
            batchsize = flist[1]
            flist = flist[0]
        
        i += 1
        nbRuns = len(flist)
        print name, nbRuns, 'runs',
        
        if targetcutoff != None and onlysuccessful:
            # filter out unsuccessful runs
            flist = filter(isSuccessful, flist)
            print ',', len(flist), 'of which were successful.'
            if len(flist) == 0:
                continue
            flist = map(relevantPart, flist)
        else:
            print
        
        longestRun = max(map(len, flist))
        xAxis = array(range(longestRun))*batchsize
    
        summed = zeros(longestRun)
        for l in flist:
            summed += paddedClipped(l, longestRun)
        yPlot = paddedClipped(summed / len(flist), longestRun)
               
        if semilog:
            semilogy()
        
        if varyplotsymbols:
            psymbol = plotsymbols[i%len(plotsymbols)]
        else:
            psymbol = '-'
        
        plot(xAxis, yPlot, psymbol, label = name)
        
    ylabel('-fitness')
    xlabel('number of evaluations')
    pylab.title(title)
    legend()
    if show:
        pylab.show()
    