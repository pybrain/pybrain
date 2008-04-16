__author__ = 'Tom Schaul, tom@idsia.ch'


import os
from scipy import array, reshape

from nesexperiments import pickleReadDict

filedir = '../temp/cartpole/'
allfiles = os.listdir(filedir)
alldata = {}
for f in allfiles:
    if f[-12:] == 'small.pickle':
        runid = f[-17:-12]
        runtype = f[:-19]
        allfits = pickleReadDict('../temp/cartpole/'+f[:-7])
        if runtype not in alldata:
            alldata[runtype] = []
        alldata[runtype].append(allfits)

for runtype, res in sorted(alldata.items()):
    print 'Experiment type:', runtype
    print 'Number of runs:', len(res)
    successes = filter(lambda x: len(x) > 0 and max(x) >= 50000, res)
    failures = filter(lambda x: len(x) > 0 and max(x) < 50000, res)
    print 'Successful runs:', len(successes), '(', int(len(successes)/float(len(res))*1000)/10.0, '%)'
    if len(successes) > 0:
        print 'Average number of episodes until success (not considering failures):',  sum(map(len, successes))/float(len(successes))
    if len(failures) > 0:
        print 'Average fitness (among failures):',  sum(map(max, failures))/float(len(failures))
    print
    print 'Number of episodes for all runs', map(len, res)
    if len(successes) > 0:
        print 'Number of episodes for successful runs', map(len, successes)
    failures = filter(lambda x: len(x) > 0 and max(x) < 100000, res)
    if len(failures) > 0:
        print '(best fitness, nb of episodes) for failed runs:', zip(map(max, failures),map(len, failures))
    print '-'*70
    print
    
            
            