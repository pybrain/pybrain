__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import array, log10, zeros, ones, power
import pylab

from nesexperiments import mainfunctions, otherfunctions
from nesexperiments import ExperimentalData, NESExperiment


dim = 15

lines = ['-', ':', '-.','--', '.-', 'x',',-', '1-', '2','-']

if __name__ == '__main__':
    line = 0
    rundata = {}
    folder = NESExperiment.folder
    for f in mainfunctions+otherfunctions:
        name = f.__name__+str(dim)+'-uni'
        rundata[f] = ExperimentalData.load(folder+name+'-b')        
    
    
    # a comparative plot, with averaging:
    prelog = False
    pylab.figure()   
        
        
    #for D in rundata.values():
    #    for id in D.expids.keys():
    #        if not id in D.fullResults:
    #            continue
            
    for D in rundata.values():
        for id in D.expids.keys():
            if not id in D.fullResults:
                continue
            line += 1
            cma = D.expids[id]['cma']
            if not cma:
                name = id[:-4]
                
            
            lower = -6
            if id[:2] == 'Sh' or id[:2] == 'Pa':
                lower = -2
            
            if 'stopPrecision' in D.fullResults[id][0]:
                lower = log10(D.fullResults[id][0]['stopPrecision'])
    
            mgens = 0
            for r in D.fullResults[id]:
                if r['success'] == True:
                    mgens = max(mgens, r['gens'])
            #mgens -= 1
            xs = array(range(mgens+1))*r['evalsPerGen']    
                
            ys = zeros(mgens+1)
            c = 0
            for r in D.fullResults[id]:
                if r['success'] == True:
                    c += 1
                    if prelog:
                        tmp = log10(array(r['xVals']).flatten())
                        if len(tmp) > mgens+1:
                            tmp = tmp[:mgens+1]
                        ys[0:len(tmp)] += map(lambda x: max(x, lower), tmp)
                        ys[len(tmp):] += lower*ones(mgens+1-len(tmp))
                    else:
                        tmp = array(r['xVals']).flatten()
                        if len(tmp) > mgens+1:
                            tmp = tmp[:mgens+1]
                        ys[0:len(tmp)] += map(lambda x: max(x, lower), tmp)
                        ys[len(tmp):] += 10**(lower)*ones(mgens+1-len(tmp))
                        
            ys /= c  
            if cma:
                tag = 'CMA'+id
                line = '-'
            else:
                if dim >= 10:
                    tag = id[:-14]
                else:
                    tag = id[:-13]                
                
            if prelog:
                ys = power(10, ys)
                pylab.semilogy()
                pylab.plot(xs, ys, lines[line], label=tag)  
            else:
                pylab.semilogy()
                pylab.plot(xs, ys, lines[line], label=tag)  
        
            if not cma:
                # print some info
                print name
                print 'runs', len(D.fullResults[id])
                print 'successes', c
                for p in ['lambd', 'lr', 'lrSigma', 'ranking', 'maxEvals', 'stopPrecision']:
                    if p in D.fullResults[id][0]:
                        print p, D.fullResults[id][0][p]
        
    if True:
        name = 'unimodal-'+str(dim)
        pylab.title(name)    
        pylab.ylabel('cost')
        pylab.xlabel('number of evaluations')
        pylab.legend()
        pylab.savefig(folder+name+'.eps')
    
    pylab.show()
                
        