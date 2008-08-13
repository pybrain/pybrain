""" This script is supposed to determine statistics on the playing 
performance of trained CaptureGameNetworks, compared on different board size."""

__author__ = 'Tom Schaul, tom@idsia.ch'

from randomsearchnetworks import randEval
from nesexperiments import pickleDumpDict, pickleReadDict
from spectator import  readNetAndParams
from resultreader import getTaggedFiles, selectSome
        

if __name__ == '__main__':
    # settings
    tag = 'x-'
    trainsize = 7
    dirout = '../temp/stats/'
    dirin = '../temp/capturegame/'
    minGeneration = 80
    maxGeneration = 200
    minScore = -1
    
    capturegame = True
    killer = True
    coevolution = False
    
    if trainsize == 5:
        sizes = [5,9]
    else:
        sizes = [trainsize,trainsize+4]
    
    
    if coevolution:
        if capturegame:
            dirin += '1/'
        else:
            dirin += '2/'
    else:
        if capturegame:
            dirin += '3/'
        else:
            dirin += '4/'
    
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
    if not coevolution:
        stype += '-es'
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
    selected = selectSome(files, ['-s'+str(trainsize),
                                  ], requireAll = True)
    
    for fname in selected:
        info = fname[len(dirin):]
        print
        print 'Reading from', info
        
        if info not in results:
            results[info] = {}
        
        net, all = readNetAndParams('', fname)
        print 'found:', len(all)
        if len(all) <= minGeneration:
            continue
        
        
        for gen, (partscore, w) in enumerate(all[minGeneration:min(maxGeneration, len(all))]):
            if partscore < minScore:
                print 'low score'
                continue
            
            wt = tuple(w)
            if wt in results[info]:
                print 'existing.'
                continue
            
            gen += minGeneration
            print 'Generation', gen
            
            scores = []
            for bsize in sizes:
                if bsize == trainsize and tag in ['we1-', 'we2-']:
                    score = partscore
                    print 'Size', bsize, ':', score
                else:
                    if bsize == trainsize:                    
                        moreEvals = 60
                    else:
                        moreEvals = 100
                        
                    if coevolution:
                        nmc = 0
                    else:
                        nmc = 0.2
                    score = randEval(size = bsize, hsize = net.hsize, setParams = w, avgOver = moreEvals, numMovesCoeff = nmc)
                    
                    if bsize == trainsize:
                        score *= 0.6 
                        score += partscore * 0.4
                        print '+', partscore, '=', score            
                scores.append(score)
                
            results[info][wt] = (zip(sizes, scores), w, gen)
            print ':',
            pickleDumpDict(fnameout, results)
            print '.'
        
#
#    # plot the results
#    if plotting or onlyplot:
#        import pylab
#        
#        if True:
#            for othersize in sizes:
#                if othersize == trainsize:
#                    continue
#                pylab.figure()
#                pylab.plot([-1,1], [-1,1], '.')
#                title = stype+' train:'+str(trainsize)+' test:'+str(othersize)
#                pylab.title(title)
#                for info in results.keys():
#                    # for now, only plot the first and the last size against each other
#                    xs, ys = [], []
#                    for (res, w, gen) in results[info].values():
#                        x, y = None, None
#                        for s, val in res:
#                            if s == othersize:
#                                x = val
#                            elif s == trainsize:
#                                y = val                        
#                        if x != None and y != None:
#                            xs.append(x)
#                            ys.append(y)
#                    if len(xs) > 1:
#                        print othersize, trainsize, 'params', info
#                        print 'samples', len(xs), 
#                        print 'correlation:', pearsonr(xs, ys)[0]
#                        pylab.plot(xs, ys, '.', label = info)
#                pylab.legend()    
#                pylab.savefig(dirout+title+'.eps')
#                  
#        pylab.show()
#        