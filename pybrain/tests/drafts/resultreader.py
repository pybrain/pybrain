""" Utilities to get results produced by experiments. """

__author__ = 'Tom Schaul, tom@idsia.ch'

import os
import pylab
from scipy import array, zeros

from pybrain.tools.xml import NetworkReader


def getTaggedFiles(dir, tag, extension = '.pickle'):
    """ return a list of all files in the specified directory
    with a name stating with the given tag (and the specified extension). """
    allfiles = os.listdir(dir)
    res = []
    for f in allfiles:
        if f[-len(extension):] == extension and f[:len(tag)] == tag:
            res.append(dir+f)
    return res
    
def selectSome(strings, requiredsubstrings = [], requireAll = False):    
    """ Filter the list of strings to only contain those that have at least 
    one of the required substrings. """
    if len(requiredsubstrings) == 0:
        return strings
    res = []
    for s in strings:
        if requireAll:
            bad = False
            for rs in requiredsubstrings:
                if s.find(rs) < 0:
                    bad = True
                    break
            if not bad:
                res.append(s)
        else:
            for rs in requiredsubstrings:
                if s.find(rs) >= 0:
                    res.append(s)
                    break
    return res

def slidingAverage(a, avgOver = 5):
    if isinstance(a, list):
        a = array(a)
    if avgOver > len(a):
        avgOver = len(a)/2
    res = zeros(len(a) - (avgOver-1))
    for i in range(avgOver):
        res += a[i:(len(a) - avgOver +i+1)]
    res /= avgOver
    return res
    
    
if __name__ == '__main__':
    dir = '../temp/capturegame/1/'
    tag = 'p'
    ext = '.xml'
    files = getTaggedFiles(dir, tag, ext)
    numPops = 2
    avgOver = 5
    plotrelative = True
    selected = selectSome(files, [#'',
                                  #'7004',
                                  #'8283',
                                  'Compe',
                                  #'MultiPop'+str(numPops)
                                  ],  requireAll = True)
    print selected
    nets = []
    otherdata = {}
    for f in selected:
        n = NetworkReader.readFrom(f)
        nets.append(n) 
        print f
        print n.name
        n._fname = f
        if hasattr(n, '_unknown_argdict'):
            otherdata[n] = n._unknown_argdict.copy()
            del n._unknown_argdict
            for k, val in otherdata[n].items():
                pass
        print
    
    hm = ['-', '.-', 'o', '.', ':']
    
    for n in nets:
        if 'RUNRES' in otherdata[n]:
            absfits = otherdata[n]['RUNRES']
            pylab.figure()
            pylab.title('Absolute '+n.name+' averaged '+str(avgOver))
            pylab.xlabel(n._fname)
            for g in range(numPops):
                popfits = absfits[g::numPops]
                pylab.plot(range(avgOver/2, len(popfits) - avgOver/2),
                           slidingAverage(popfits, avgOver), 
                           hm[g%numPops], label = 'avg'+str(g+1))                
                pylab.plot(popfits, 
                           hm[g%numPops], label = 'abs'+str(g+1))                
            pylab.legend()
            
        if plotrelative and 'HoBestFitnesses' in otherdata[n]:
            relfits = otherdata[n]['HoBestFitnesses']
            avgRelFits = map(lambda x: sum(x)/len(x), relfits)
            bestRelFits = map(max, relfits)
            pylab.figure()
            pylab.title('Relative '+n.name+' averaged '+str(avgOver))
            for g in range(numPops):
                pylab.plot(slidingAverage(avgRelFits[g::numPops], avgOver), 
                           hm[g%numPops], label = 'avg'+str(g+1))
                pylab.plot(slidingAverage(bestRelFits[g::numPops], avgOver), 
                           hm[g%numPops], label = 'max'+str(g+1))
            pylab.legend()
    if len(selected) > 0:
        pylab.show()
                