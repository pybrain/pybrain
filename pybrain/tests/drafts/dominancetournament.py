from pybrain.structure.evolvables.cheaplycopiable import CheaplyCopiable
__author__ = 'Tom Schaul, tom@idsia.ch'

#TODO: clean up and transfer functionality to tools

from pybrain.rl.tasks.capturegame import RelativeCaptureTask
from pybrain.tools.xml import NetworkReader
from spectator import readNetAndParams
from resultreader import getTaggedFiles, selectSome
    
    
def dominanceNumber(relEval, champions):
    dominators = [champions[0]]
    print 0
    for i, c in enumerate(champions[1:]):
        print i+1,
        beatAll = True
        for d in reversed(dominators):
            if relEval(c, d) < 0:
                beatAll = False
                break
            else:
                print '.',
        if beatAll:
            print 'Dominant!'
            dominators.append(c)
        else:
            print
    return len(dominators)
        



if __name__ == '__main__':
    dir = '../temp/capturegame/1/'
    tag = 'p'
    ext = '.xml'
    files = getTaggedFiles(dir, tag, ext)
    selected = selectSome(files, ['7004',
                                  ], requireAll = True)
    fname = selected[0][len(dir):]
    print fname
    n, all = readNetAndParams(dir, fname)
    net = CheaplyCopiable(n)
    champions = []
    for i, (s, p) in enumerate(all):
        print i, ':', s
        tmp = net.copy()
        tmp._setParameters(p)
        champions.append(tmp)
    relativeTask = RelativeCaptureTask(n.size, useNetworks = True, maxGames = 1,
                                       minTemperature = 0, presetGamesProportion = 0,
                                       )
    print dominanceNumber(relativeTask, champions)
    