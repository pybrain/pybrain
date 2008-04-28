""" get the results produced by fem-pole-experiments. """

__author__ = 'Tom Schaul, tom@idsia.ch'

import os
from scipy import array, average, sqrt, var
from nesexperiments import pickleReadDict

def readTaggedFiles(tag):
    filedir = '../temp/fem/'
    allfiles = os.listdir(filedir)
    alldata = {}
    for f in allfiles:
        if f[-7:] == '.pickle' and f[:len(tag)] == tag:
            #runid = f[-12:-7]
            runtype = f[len(tag):-13]
            
            if runtype[-5:] == '-fail':
                runtype = runtype[:-5]
                good = False
            else:
                good = True
            
            nevals = len(pickleReadDict(filedir+f[:-7])['allevals'])
            
            if runtype not in alldata:
                alldata[runtype] = [0, 0, []]
            
            alldata[runtype][0] += 1
            if good:
                alldata[runtype][1] += 1
                alldata[runtype][2].append(nevals)
            
    return alldata

if __name__ == '__main__':
    tag = 'wewe'
    print tag
    d = readTaggedFiles(tag)
    for k, val in sorted(d.items()):
        print k, 'Runs:', val[0], 'Successful', val[1], 
        avgevals = average(array(val[2]))
        print 'Average number of evaluations until success:', avgevals
        print 'Std dev:', sqrt(var(array(val[2])))
        