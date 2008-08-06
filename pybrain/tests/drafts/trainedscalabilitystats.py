""" This script is supposed to determine statistics on the playing 
performance of trained CaptureGameNetworks, compared on different board size."""

__author__ = 'Tom Schaul, tom@idsia.ch'

from randomsearchnetworks import randEval
from nesexperiments import pickleDumpDict, pickleReadDict
from scipy.stats import pearsonr
from spectator import  readNetAndParams
from resultreader import getTaggedFiles, selectSome
    
    

if __name__ == '__main__':
    # settings
    tag = 'p-'
    sizes = [5,9,
             #19,
             ]
    trainsize = 5
    dirout = '../temp/stats/'
    dirin = '../temp/capturegame/1/'
    minGeneration = 50
    minScore = -1
    minData = 0
    plotting = True
    capturegame = True
    killer = True
    onlyplot = False
    
    # build the type name
    stype = tag + 'comparative-'
    if capturegame:
        stype += 'capture'
    else:        
        stype += 'gomoku'
    if killer:
        stype += '-killer'
    else:
        stype += '-random'
    stype += '-trainOn'+str(trainsize)
    fnameout = dirout+stype
    print stype
    
    #old results: (keys are info tags: learning files)
    results = pickleReadDict(fnameout)
    
    olds = 0
    for k in results.keys():
        olds += len(results[k])
    print 'Old results:', olds, 'runs.'
    
    
    ext = '.xml'
    files = getTaggedFiles(dirin, tag, ext)
    if onlyplot:
        selected = []
    else:
        selected = selectSome(files, ['-s'+str(trainsize),
                                      ], requireAll = True)
    
    for fname in selected:
        info = fname[len(dirin):]
        if info not in results:
            results[info] = {}
        net, all = readNetAndParams('', fname)
        if len(all) <= minGeneration:
            continue
        print
        print 'Reading from', info
        
        for gen, (partscore, w) in enumerate(all[minGeneration:]):
            if partscore < minScore:
                continue
            
            wt = tuple(w)
            if wt in results[info]:
                continue
            
            gen += minGeneration
            print 'Generation', gen
            
            scores = []
            for bsize in sizes:
                if bsize == trainsize:
                    moreEvals = 60
                else:
                    moreEvals = 100
                    
                score = randEval(size = bsize, hsize = net.hsize, setParams = w, avgOver = moreEvals)
                
                if bsize == trainsize:
                    score *= 0.6 
                    score += partscore * 0.4
                    print '+', partscore, '=', score            
                scores.append(score)
                
            results[info][wt] = (zip(sizes, scores), w, gen)
            print ':',
            pickleDumpDict(fnameout, results)
            print '.'
        

    # plot the results
    if plotting or onlyplot:
        import pylab
        
        if True:
            for othersize in sizes:
                if othersize == trainsize:
                    continue
                pylab.figure()
                pylab.plot([-1,1], [-1,1], '.')
                title = stype+' train:'+str(trainsize)+' test:'+str(othersize)
                pylab.title(title)
                for info in results.keys():
                    # for now, only plot the first and the last size against each other
                    xs, ys = [], []
                    for (res, w, gen) in results[info].values():
                        x, y = None, None
                        for s, val in res:
                            if s == othersize:
                                x = val
                            elif s == trainsize:
                                y = val                        
                        if x != None and y != None:
                            xs.append(x)
                            ys.append(y)
                    if len(xs) > 1:
                        print othersize, trainsize, 'params', info
                        print 'samples', len(xs), 
                        print 'correlation:', pearsonr(xs, ys)[0]
                        pylab.plot(xs, ys, '.', label = info)
                pylab.legend()    
                pylab.savefig(dirout+title+'.eps')
                  
        pylab.show()
        