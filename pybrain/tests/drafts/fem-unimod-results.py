""" get the results produced by femexperiments. """

__author__ = 'Tom Schaul, tom@idsia.ch'

import os
import pylab    
from scipy import array, average, var, sqrt, zeros, size

from nesexperiments import pickleReadDict, pickleDumpDict

def readTaggedFiles(tag, selective = None, verbose = True):
    filedir = '../temp/fem/'
    allfiles = os.listdir(filedir)
    alldata = {}
    for f in allfiles:
        if f[-7:] == '.pickle' and f[:len(tag)+1] == tag+'-':
            runid = f[-12:-7]
            runtype = f[len(tag)+1:-13]
            if selective != None:
                if runtype not in selective:
                    continue
            
            if runtype not in alldata:
                alldata[runtype] = [0, 0, []]
            d = pickleReadDict(filedir+f[:-7])
            allfits = d['allevals']
            mufits = d['muevals']
            solved = (max(allfits)> -1e-10)
            
            alldata[runtype][2].append((solved, allfits, mufits))
            alldata[runtype][0] += 1
            if solved:
                alldata[runtype][1] += 1
            if verbose:
                print runtype, runid, ' '*8,
                if solved:
                    print 'solved in:', len(allfits)
                else:
                    print 'not soved. Best:', max(allfits), 'number of evals:', len(allfits)
            
            
    return alldata

if __name__ == '__main__':
    tag = 'good'
    print 'tag:', tag
    d = readTaggedFiles(tag, ['SphereFunction15',
                              'CigarFunction15',
                              'SchwefelFunction15',
                              'TabletFunction15',
                              'DiffPowFunction15',
                              'ElliFunction15',
                              ])
    res = {}
    for k, val in sorted(d.items()):
        print k, 'Runs:', val[0], 'Successful', val[1]
        lens = array(map(lambda x: len(x[1]), filter(lambda x: x[0], val[2])))
        if len(lens) > 0:
            print 'Average number of evaluations until success:', average(lens), 'std dev', sqrt(var(lens))
    
        # produce a dict with only lists of muevals for each case:
        tmp = []
        for so, ae, me in val[2]:
            #if so:
            tmp.append(me)
        res[k] = [val[1], lens, tmp]
        
        # plotting
        if True and len(lens) >0:
            allmuevals = map(lambda x: x[2], filter(lambda x: x[0], val[2]))
            maxlen = max(map(len, allmuevals))
            merged = zeros(maxlen)
            for me in allmuevals:
                tmp = array(me)
                merged[:size(tmp)] += tmp
            merged /= len(allmuevals)
            merged.clip(min = - 1e10, max = -1e-10)
            x = array(range(maxlen))*25
            pylab.semilogy()
            pylab.plot(x, -merged, label = k)
    
            
    pickleDumpDict('../temp/fem/dim15results-b', res)
    
    print 'written'
    
    if True:
        pylab.ylabel('fitness')
        pylab.xlabel('number of evaluations')
        pylab.legend()
        pylab.show()
    
    
    