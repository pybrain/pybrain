""" This script is supposed to determine statistics on the playing 
performance of randomly generated CaptureGameNetworks, compared on different board size."""

__author__ = 'Tom Schaul, tom@idsia.ch'

from randomsearchnetworks import randEval, iterArgumentCombinations
from nesexperiments import pickleDumpDict, pickleReadDict
from pybrain.rl.agents.capturegameplayers import KillingPlayer, RandomCapturePlayer
from pybrain.rl.agents.gomokuplayers import KillingGomokuPlayer, RandomGomokuPlayer


if __name__ == '__main__':
    # settings
    tag = 'x-'
    capturegame = True
    killer = False
    handicap = False
    sizes = [5,9,19]                
    argsVars = {'hsize': [1, 5],
                'initScaling': [1, 10],
                }
    dir = '../temp/stats/'
    repeat = 150
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
            
            tmp = []
            # first run on smallest size
            x, p = randEval(size = sizes[0], allReturn = True, **args)
            tmp.append(x)
            for s in sizes[1:]:
                tmp.append(randEval(size = s, setParams = p, **args))
                
            results[key].append((zip(sizes, tmp), p))
            print ':',
            pickleDumpDict(fname, results)
            print '.'


    # plot the results
    if plotting:
        import pylab
        pylab.plot([-1,1], [-1,1], '.')
        title = stype+' '+str(sizes[0])+' vs. '+str(sizes[-1])
        pylab.title(title)
        for k in results.keys():
            # for now, only plot the first and the last size against each other
            x, y = [], []
            for point in results[k]:
                if point[0][0][0] == sizes[0]:
                    if point[0][-1][0] == sizes[-1]:
                        x.append(point[0][0][1]) 
                        y.append(point[0][-1][1]) 
                        
            pylab.plot(x, y, '.', label = k)
        pylab.legend()    
        pylab.savefig(dir+title+'.eps')
        pylab.show()
    