""" This script is supposed to determine statistics on the playing 
performance of randomly generated CaptureGameNetworks. """

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.agents.capturegameplayers import KillingPlayer, RandomCapturePlayer
from pybrain.rl.agents.gomokuplayers import KillingGomokuPlayer, RandomGomokuPlayer
from pybrain.rl.tasks.capturegame import CaptureGameTask, HandicapCaptureTask
from pybrain.rl.tasks.gomoku import GomokuTask
from pybrain.structure.networks.custom.capturegame import CaptureGameNetwork
from pybrain.utilities import fListToString
from nesexperiments import pickleDumpDict, pickleReadDict
from pybrain import buildNetwork, TanhLayer, SigmoidLayer, MDLSTMLayer
        

def randEval(size = 0, hsize = 0, opponent = None, handicap = False, mlp = False, capturegame = True, initScaling = 1, 
             avgOver = 100, verbose = True, setParams = None, allReturn = False,
             lstm = False):
    if mlp:
        # comarison with simple MLP
        net = buildNetwork(2 * size**2, hsize * size**2, size**2, 
                           hiddenclass = TanhLayer, outclass = SigmoidLayer)        
    else:
        if lstm:
            net = CaptureGameNetwork(size = size, hsize = hsize, simpleborders = True,
                                     componentclass = MDLSTMLayer)         
        else:
            net = CaptureGameNetwork(size = size, hsize = hsize, simpleborders = True)         
    if setParams != None:
        net._params[:] = setParams
    else:
        net.randomize()
        net._params /= initScaling # start with small values?    
    
    if capturegame:
        if handicap:
            handicapTask = HandicapCaptureTask(size, opponent = opponent)
            res = handicapTask(net)
        else:
            absoluteTask = CaptureGameTask(size, averageOverGames = avgOver, alternateStarting = True, 
                                           opponent = opponent)
            res = absoluteTask(net)
    else:
        # Gomoku#            del results[k]
    #    pickleDumpDict(fname, results)
    
        absoluteTask = GomokuTask(size, averageOverGames = avgOver, alternateStarting = True, 
                                  opponent = opponent)
        res = absoluteTask(net)
                    
    if verbose:
        print 'Size', size, 'H', hsize, 'res', int(res*1000)/1000.,
        if initScaling != 1:
            print 'initScaling', initScaling,
        if net.paramdim < 20:
            print fListToString(net.params, 4)
        else:
            print
            
    if allReturn:
        return res, net.params
    else:
        return res

def iterArgumentCombinations(d):
    """ Returns dictionnaries with all argument variations. 
    Input is a dictionary of lists. """
    res = {}
    keys = sorted(d.keys())
    key = keys[0]
    if len(keys) == 1:
        for v in d[key]:
            res[key] = v
            yield res
    else:
        tmp = d.copy()
        del tmp[key]
        for args in iterArgumentCombinations(tmp):
            res = args.copy()
            for v in d[key]:
                res[key] = v
                yield res
            
        
if __name__ == '__main__':
    # settings
    tag = 'x-'
    capturegame = False
    killer = True
    handicap = False
    mlp = False
    argsVars = {'size': [7,11],
                'hsize': [5,10],
                'initScaling': [1,10],
                }
    dir = '../temp/stats/'
    repeat = 1
    minData = 10
    plotting = True
    
    # build the type name
    if capturegame:
        stype = tag+'capture'
    else:        
        stype = tag+'gomoku'
    if killer:
        stype += '-killer'
    else:
        stype += '-random'
    if handicap:
        stype += '-handicap'
    if mlp:
        stype += '-mlp'
    fname = dir+stype
    print stype
    
    #old results:
    results = pickleReadDict(fname)
    otherresults = pickleReadDict(dir+tag+'comparative-'+stype[len(tag):])
    olds = 0
    for k in results.keys():
        olds += len(results[k])
    for k in otherresults.keys():
        olds += sum(map(len, otherresults[k]))        
    print 'Old results:', olds, 'runs.'
    
    for i in range(1):
        # produce new results
        for args in iterArgumentCombinations(argsVars):
            key = (args['size'], args['hsize'], args['initScaling'])
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
            args['mlp'] = mlp
            try:
                results[key].append(randEval(**args))
                print ':',
                pickleDumpDict(fname, results)
                print '.'
            except:
                print 'Oh-oh'
        
    # print a summary
    if not plotting:
        for k in sorted(results.keys()):
            val = results[k]
            print k, int(sum(val)/float(len(val))*1000)/1000., len(val)
            print fListToString(val, 2)
            print
            
    # plot the results (according to percentiles/performance)
    else:
        # TODO: outsource percentile-plotting
        import pylab
        from scipy import array
        for k in sorted(results.keys()):
            val = results[k]
            k2 = tuple(k[1:])
            if k2 in otherresults:
                for point in otherresults[k2]:
                    for dim, score in point[0]:
                        if dim == k[0]:
                            val.append(score)
            print k, k2, len(val)
            if len(val) < minData:
                continue
            x = array([max(val)]+sorted(val)[::-1])
            y = array(map(float, range(len(x))))
            y /= (len(x)-1)
            pylab.plot(y, x, label = str(k)+'-'+str(len(x)-1))
        pylab.legend()
        pylab.title(stype)
        pylab.savefig(dir+stype+'.eps')
        pylab.show()
    