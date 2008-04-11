__author__ = 'Tom Schaul, tom@idsia.ch'


import pylab
from scipy import array, zeros, var, sqrt

from rwrexperiments import ExperimentalData, RwrExperiment


#name = 'CheeseMaze-3941'
#name = 'CheeseMaze-4349'
#name = 'CheeseMaze-6443'
#name = 'CheeseMaze-8248'
#name = 'CheeseMaze-9108'

#name = 'FourByThreeMaze-4942'
#name = 'FourByThreeMaze-6907'
#name = 'FourByThreeMaze-7128'
#name = 'FourByThreeMaze-9367'

#name = 'ShuttleDocking-5725'
#name = 'ShuttleDocking-5847'
#name = 'ShuttleDocking-6799'
#name = 'ShuttleDocking-8783'

#name = 'TigerTask-1883'
#name = 'TigerTask-6732'
#name = 'TigerTask-9859'
#name = 'TigerTask-9952'

#name = 'TMaze3-5789'
#name = 'TMaze3-8104'

#name = 'TMaze5-1854'
#name = 'TMaze5-6754'

#name = 'TMaze7-1077'
#name = 'TMaze7-6174'
#name = 'TMaze7-8099'

#name = 'TMaze10-6450'
#name = 'TMaze10-7448'

allplots = False
selective = True

if __name__ == '__main__':
    skipped = 0
    filename = RwrExperiment.folder+name
    E = ExperimentalData.load(filename)
    for id in E.expids.keys():
        print id
        if not id in E.fullResults:
            continue
        allAvgRewards = []
        allGreedys = []
        for info in E.fullResults[id]:
            res = info['rewardAvg']
            if selective and res[0] > res[-1]:
                skipped += 1
                continue
            allAvgRewards.append(res)
            allGreedys.append(info['greedyAvg'])
            if allplots:
                x = array(range(len(res)))*info['params']['batchSize']
                pylab.plot(x, array(info['rewardAvg']))
        
        print info['params']
        repeat = len(allAvgRewards)
        few = 3
        if repeat <= 0:
            continue
        res = zeros(len(allAvgRewards[0]))
        lasts = zeros(repeat*few)
        firsts = zeros(repeat)
        for i, ar in enumerate(allAvgRewards):
            res += array(ar)
            firsts[i] = ar[0]
            lasts[i*few:i*few+few] = ar[-few:]
        res /= repeat
        print 'Random:', sum(firsts)/(repeat), '+-', sqrt(var(firsts))
        print 'Stoch:', repeat, sum(lasts)/(repeat*few), '+-', sqrt(var(lasts))
        x = array(range(len(allAvgRewards[0])))*info['params']['batchSize']
        pylab.plot(x, res, 'r-')
        
        res = zeros(len(allGreedys[0]))
        lasts = zeros(repeat*few)
        for i, ar in enumerate(allGreedys):
            res += array(ar)
            lasts[i*few:i*few+few] = ar[-few:]
            
        res /= repeat
        print 'Greedy:', repeat, sum(lasts)/(repeat*few), '+-',sqrt(var(lasts))
        pylab.plot(x, res, 'b-.')
    print 'Skipped', skipped
    pylab.ylabel('average reward')
    pylab.xlabel('number of episodes')
    pylab.savefig(filename+'.eps')
    pylab.show()
        