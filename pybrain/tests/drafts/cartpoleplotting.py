__author__ = 'Tom Schaul, tom@idsia.ch'


import os, pylab
from scipy import array, reshape

from nesexperiments import pickleReadDict

filedir = '../temp/cartpole2/'
allfiles = os.listdir(filedir)
alldata = {}
for f in allfiles:
    if f[-7:] == '.pickle':
        runid = f[-12:-7]
        runtype = f[:-14]
        tmp = pickleReadDict('../temp/cartpole/'+f[:-7])
        if len(tmp) >= 10000:
            if runtype not in alldata:
                alldata[runtype] = {}
            alldata[runtype][runid] = tmp
            fits = array(map(lambda x:x[1], tmp)[:10000])
            block = reshape(fits, (100, 100))
            avgs = array(map(lambda x: sum(x), block))/100.
            print runtype, runid, len(tmp), max(avgs)
            pylab.plot(avgs,
                       #label = runtype
                       )            
#pylab.legend()
pylab.savefig(filedir+'summary.eps')
pylab.show()
            

