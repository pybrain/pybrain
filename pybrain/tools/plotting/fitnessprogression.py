""" a type of plots used so frequently that I think they merit their own utility """

__author__ = 'Tom Schaul, tom@idsia.ch'

import pylab
from pylab import xlabel, ylabel, legend, plot, semilogy
from scipy import array, zeros, power, log10
from pybrain.utilities import avgFoundAfter

plotsymbols = ['-', ':', '-.']
psymbol = '-'


def plotFitnessProgession(fitdict, batchsize=1, semilog=True,
                          targetcutoff=1e-10, minimize=True,
                          title=None, verbose=True,
                          varyplotsymbols=False,
                          averageOverEvaluations=True,
                          onlysuccessful=False,
                          useMedian=False,
                          resolution=1000):
    """ Plot multiple fitness curves on a single figure, with the following customizations:

        :arg fitdict: a dictionary mapping a name to a list of fitness-arrays
        :key batchsize: the number of evaluations between two points in fitness-arrays
                          specific batch sizes can also be given given in fitdict
        :key targetcutoff: this gives the cutoff point at the best fitness
        :key averageOverEvaluations: averaging is done over fitnesses (for a given number of evaluations)
                                    or over evaluations required to reach a certain fitness.
        :key resolution: resolution when averaging over evaluations
        :key onlysuccessful: consider only successful runs
        :key title: specify a title.
        :key varyplotsymbols: used different line types for each curve.
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
            res[:len(l)] += l.clip(min=targetcutoff, max=1e100)
        else:
            res[:len(l)] += l.clip(max=targetcutoff, min= -1e100)
        return res

    def relevantPart(l):
        """ the part of the vector that's above the cutoff. """
        if targetcutoff != None:
            for i, val in enumerate(l):
                if minimize and val <= targetcutoff:
                    return l[:i + 1]
                elif not minimize and val >= targetcutoff:
                    return l[:i + 1]
        return l



    i = 0
    for name, flist in sorted(fitdict.items()):
        if isinstance(flist, tuple):
            batchsize = flist[1]
            flist = flist[0]

        i += 1
        nbRuns = len(flist)
        print(name, nbRuns, 'runs',)

        if targetcutoff != None:
            if onlysuccessful:
                # filter out unsuccessful runs
                flist = filter(isSuccessful, flist)
                print(',', len(flist), 'of which were successful.')
            else:
                print
            # cut off irrelevant part
            flist = map(relevantPart, flist)

        if len(flist) == 0:
            continue

        if averageOverEvaluations:
            worstPerf = max(map(max, flist))
            if semilog:
                yPlot = list(reversed(power(10, ((array(range(resolution + 1)) / float(resolution)) *
                                             (log10(worstPerf) - log10(targetcutoff)) + log10(targetcutoff)))))
            else:
                yPlot = list(reversed((array(range(resolution + 1)) / float(resolution)) *
                                             (worstPerf - targetcutoff) + targetcutoff))
            xPlot = avgFoundAfter(yPlot, flist, batchsize, useMedian=useMedian)

        else:
            longestRun = max(map(len, flist))
            xPlot = array(range(longestRun)) * batchsize
            summed = zeros(longestRun)
            for l in flist:
                summed += paddedClipped(l, longestRun)
            yPlot = paddedClipped(summed / len(flist), longestRun)

        if semilog:
            semilogy()

        if varyplotsymbols:
            psymbol = plotsymbols[i % len(plotsymbols)]
        else:
            psymbol = '-'

        plot(xPlot, yPlot, psymbol, label=name)

    ylabel('-fitness')
    xlabel('number of evaluations')
    pylab.title(title)
    legend()

