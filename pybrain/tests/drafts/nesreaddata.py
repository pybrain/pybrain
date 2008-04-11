__author__ = 'Tom Schaul, tom@idsia.ch'

import pylab
from scipy import array, log10, zeros, ones, power

from nesexperiments import ExperimentalData, NESExperiment


name = 'SchwefelFunction15-simple'
#name = 'AckleyFunction5-simple'
#name = 'WeierstrassFunction15-simple'
#name = 'ParabRFunction5-248'


if __name__ == '__main__':
    folder = NESExperiment.folder
    D = ExperimentalData.load(folder+name+'-b')
    
    for id in D.expids.keys():
        if not id in D.fullResults:
            break
        pylab.figure()   
        pylab.title(id)    
        for r in D.fullResults[id]:
            gens = r['gens']
            xs = array(range(gens))*r['evalsPerGen']
            ys = r['xVals']
            pylab.semilogy()
            pylab.plot(xs, ys)  
    
    # now a comparative plot, with averaging:
    prelog = True
    pylab.figure()   
    for id in D.expids.keys():
        if not id in D.fullResults:
            break
        cma = D.expids[id]['cma']
        if not cma:
            name = id[:-4]
        lower = -6
        if 'stopPrecision' in D.fullResults[id][0]:
            lower = log10(D.fullResults[id][0]['stopPrecision'])
            
        mgens = 0
        for r in D.fullResults[id]:
            if r['success'] == True or id == 'TabletFunction5-335':
                mgens = max(mgens, r['gens'])
            
        if id == 'SphereFunction3-707':
            mgens = 115
        
        xs = array(range(mgens))*r['evalsPerGen']    
        ys = zeros(mgens)
        c = 0
        for r in D.fullResults[id]:
            if r['success'] == True or id == 'TabletFunction5-335':
                c += 1
                if prelog:
                    tmp = log10(array(r['xVals']).flatten())
                    if len(tmp) > mgens:
                        tmp = tmp[:mgens]
                    ys[0:len(tmp)] += map(lambda x: max(x, lower), tmp)
                    ys[len(tmp):] += lower*ones(mgens-len(tmp))
                else:
                    tmp = array(r['xVals'])
                    tmp.resize(mgens)
                    ys += tmp
        ys /= c
        if cma:
            tag = 'CMA'
            line = 'b-'
        else:
            tag = 'NES'
            line = 'r.-'
        
        if prelog:
            ys = power(10, ys)
            pylab.semilogy()
            pylab.plot(xs, ys, line, label=tag)  
        else:
            pylab.semilogy()
            pylab.plot(xs, ys, line, label=tag)  
    
        if not cma:
            # print some info
            print name
            print 'runs', len(D.fullResults[id])
            print 'successes', c
            for p in ['lambd', 'lr', 'lrSigma', 'ranking', 'maxEvals', 'stopPrecision']:
                if p in D.fullResults[id][0]:
                    print p, D.fullResults[id][0][p]
    
    if True:
        pylab.title(name)    
        pylab.ylabel('-fitness')
        pylab.xlabel('number of evaluations')
        pylab.legend()
        pylab.savefig(folder+name+'.eps')
    
    pylab.show()
                
        