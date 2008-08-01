from pybrain.structure.evolvables.cheaplycopiable import CheaplyCopiable
__author__ = 'Tom Schaul, tom@idsia.ch'

#TODO: clean up and transfer functionality to tools

from pybrain.rl.tasks.capturegame import RelativeCaptureTask
from spectator import readNetAndParams
from resultreader import getTaggedFiles, selectSome
from pybrain.utilities import fListToString
    
    
def dominanceNumber(relEval, champions, verbose = True):
    dominators = [(champions[0], 0)]
    if verbose:
        print 0
    for i, c in enumerate(champions[1:]):
        if verbose:
            print i+1,
        beatAll = True
        for d, g in reversed(dominators):
            if relEval(c, d) < 0:
                beatAll = False
                break
            elif verbose:
                print '.',
        if beatAll:
            if verbose:
                print 'Dominant!'
            dominators.append((c, i+1))
        elif verbose:
            print
    return dominators
        


if __name__ == '__main__':
    dir = '../temp/capturegame/1/'
    tag = 'p'
    ext = '.xml'
    verbose = False
    files = getTaggedFiles(dir, tag, ext)
    selected = selectSome(files, [], requireAll = True)
    for fname in selected:
        fname = fname[len(dir):]
        n, all = readNetAndParams(dir, fname)
        net = CheaplyCopiable(n)
        champions = []
        for i, (s, p) in enumerate(all):
            #print i, ':', s
            tmp = net.copy()
            tmp._setParameters(p)
            champions.append(tmp)
        relativeTask = RelativeCaptureTask(n.size, useNetworks = True, maxGames = 1,
                                           minTemperature = 0, presetGamesProportion = 0)
        def relEval(x, y):
            # symmetric evaluation
            return relativeTask(x,y)-relativeTask(y,x)
        print fname
        print 'Total of', len(champions), 'generation champions'
        dchamps = dominanceNumber(relEval, champions, verbose = verbose)
        domgens = map(lambda x: x[1], dchamps)
        print 'Dominance number:', len(dchamps)
        print 'Dominant champions in generations'
        print domgens
        print 'Absolute fitnesses of dominant champions'
        print fListToString(map(lambda g: all[g][0], domgens), 2)
        print
        
        