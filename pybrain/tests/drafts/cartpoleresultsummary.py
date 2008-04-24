__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import array, zeros
import os

from nesexperiments import pickleReadDict

def readAllRunInfo():
    filedir = '../temp/cartpole3/'
    allfiles = os.listdir(filedir)
    alldata = {}
    for f in allfiles:
        if f[-7:] == '.pickle':
            runid = f[-12:-7]
            runtype = f[:-14]
            allfits = pickleReadDict(filedir+f[:-7])['allfits']
            if runtype not in alldata:
                alldata[runtype] = []
            alldata[runtype].append(allfits)
    return alldata


def summarizeRunInfo(alldata):
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
        
        
def plotBestAvgFitnessByEpisode(alldata):
    import pylab
    for runtype, res in alldata.items():
        plotrange = array(range(max(map(len, res))/100))*100
        averages = zeros(len(plotrange))
        averageOver = zeros(len(plotrange))
        for r in res:
            for i, p in enumerate((array(range(len(r)/100))*100)):
                if len(r) >= p:
                    best = max(r[:p+1])
                    averages[i] += best
                    averageOver[i] += 1.
        res = []
        for i, r in enumerate(averages):
            if averageOver[i] > 0:
                res.append(averages[i]/averageOver[i])
        pylab.title(runtype)
        pylab.figure()
        pylab.plot(res)
    pylab.show()
            
        
if __name__ == '__main__':
    info = readAllRunInfo()
    summarizeRunInfo(info)
    plotBestAvgFitnessByEpisode(info)
                
                