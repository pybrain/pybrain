__author__ = 'Tom Schaul, tom@idsia.ch'

import os

def readRunSummary(tag):
    filedir = '../temp/fem/'
    allfiles = os.listdir(filedir)
    alldata = {}
    for f in allfiles:
        if f[-7:] == '.pickle' and f[:8+len(tag)] == tag+'-multi--':
            #runid = f[-12:-7]
            runtype = f[8+len(tag):-13]
            if runtype[-2:] == '-S':
                runtype = runtype[:-2]
                good = True
            else:
                good = False
            if runtype not in alldata:
                alldata[runtype] = [0, 0]
            alldata[runtype][0] += 1
            if good:
                alldata[runtype][1] += 1
    return alldata

if __name__ == '__main__':
    #tag = 'cma'
    tag = 'ok'
    print tag
    d = readRunSummary(tag)
    for k, val in sorted(d.items()):
        print k, val, 'score:', val[1]/float(val[0])*100, '%'