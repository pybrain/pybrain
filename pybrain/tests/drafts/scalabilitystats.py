""" This script is supposed to determine statistics on the playing 
performance of randomly generated CaptureGameNetworks, compared on different board size."""

__author__ = 'Tom Schaul, tom@idsia.ch'

from randomsearchnetworks import randEval, iterArgumentCombinations
from nesexperiments import pickleDumpDict, pickleReadDict
from pybrain.rl.agents.capturegameplayers import KillingPlayer, RandomCapturePlayer
from pybrain.rl.agents.gomokuplayers import KillingGomokuPlayer, RandomGomokuPlayer
from scipy.stats import pearsonr


if __name__ == '__main__':
    # settings
    tag = 'x-'
    capturegame = False
    killer = True
    
    if capturegame:
        sizes = [5,9,19]
    else:
        sizes = [5,7,9,11]
    
    handicap = False
    lstm = False
    argsVars = {'hsize': [5],
                'initScaling': [1],
                }
    if lstm:
        tag = 'lstm-'
        argsVars['lstm'] = [True]
        argsVars['avgOver'] = [40]
                
    dir = '../temp/stats/'
    repeat = 250
    minData = 0
    plotting = True
    
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
    if handicap:
        stype += '-handicap'
    fname = dir+stype
    print stype
    
    #old results:
    results = pickleReadDict(fname)
    
    olds = 0
    for k in results.keys():
        olds += len(results[k])
    print 'Old results:', olds, 'runs.'
        
    for i in range(repeat):
        # produce new results
        for args in iterArgumentCombinations(argsVars):
            key = (args['hsize'], args['initScaling'])
            if key not in results:
                results[key] = []
            args['capturegame'] = capturegame
            if capturegame:
                if killer:
                    args['opponent'] = KillingPlayer
                else:
                    args['opponent'] = RandomCapturePlayer
            else:                
                if killer:
                    args['opponent'] = KillingGomokuPlayer
                else:
                    args['opponent'] = RandomGomokuPlayer
            args['handicap'] = handicap
            print args
            tmp = []
            # first run on smallest size
            try:
                x, p = randEval(size = sizes[0], allReturn = True, **args)
                tmp.append(x)
                for s in sizes[1:]:
                    tmp.append(randEval(size = s, setParams = p, **args))
                
                results[key].append((zip(sizes, tmp), p))
                print ':',
                pickleDumpDict(fname, results)
                print '.'
            except:
                print 'Oh-oh.'


    # plot the results
    if plotting:
        import pylab
        
        if True:
            for i in range(len(sizes)-1):
                smin = sizes[i]
                smax = sizes[i+1]
                pylab.figure()
                pylab.plot([-1,1], [-1,1], '.')
                title = stype+' '+str(smin)+' vs. '+str(smax)
                pylab.title(title)
                for k in results.keys():
                    # for now, only plot the first and the last size against each other
                    xs, ys = [], []
                    for point in results[k]:
                        x, y = None, None
                        for s, val in point[0]:
                            if s == smin:
                                x = val
                            elif s == smax:
                                y = val                        
                        if x != None and y != None:
                            xs.append(x)
                            ys.append(y)
                    if len(xs) > 1:
                        print smin, smax, 'params', k, 'samples', len(xs), 
                        print 'correlation:', pearsonr(xs, ys)
                        pylab.plot(xs, ys, '.', label = k)
                pylab.legend()    
                pylab.savefig(dir+title+'.eps')
                        
        
        if False:
            pylab.figure()
            pylab.title('(border weight * output weight) vs performance')
            xs, ys = [], []
            for point in results[(1,1)]:
                if not point[0][0][0] == 5:
                    continue
                xs.append(point[0][0][1])
                ys.append(point[1][0] * point[1][-1])
            pylab.plot(xs, ys, '.')
        
                 
        pylab.show()
        