""" A little tool for watching games """

__author__ = 'Tom Schaul, tom@idsia.ch'

import sys
import logging

from pybrain.utilities import fListToString
from nesexperiments import pickleReadDict
from pybrain.tools.xml import NetworkReader
from randomsearchnetworks import randEval


def arena(game, p1, p2, steps = True):
    """ have p1 play against p2 on the given game. """
    p1.color = game.BLACK
    p2.color = -p1.color
            
    game.reset()
    i = 0
    players = [p1, p2]
    while not game.gameOver():
        p = players[i]
        if steps and i == 0:
            r = sys.stdin.readline()
            print r
            
        game.performAction(p.getAction())
        i = (i+1)%2
        if steps and i == 0:
            print game
            if isinstance(p2, ModuleDecidingPlayer):
                o = p2.module.outputbuffer[0]
                s = game.size
                for j in range(s):
                    logging.info(fListToString(o[s*j:s*(j+1)], 4))
    if not steps:
        logging.info(game)

def getBestAgent(dir, fname, size, dkey, winningindex = 0):
    """ read the parameters of the best-scoring individual on the specified size. """
    results = pickleReadDict(dir+fname)
    all = []
    for point in results[dkey]:
        for dim, val in point[0]:
            if size == None or dim == size:
                all.append((val, point[1], dim))
    all.sort(key = lambda x: -x[0])
    i = min(len(all)-1, winningindex)
    bval, best, size = all[i]
    logging.info('Evaluated Size: '+str(size))
    logging.info('HiddenSize: '+str(dkey[0])+' initScaling: '+str(dkey[1]))
    logging.info('Score: '+str(bval))
    logging.info('Item '+str(i+1)+' of '+str(len(all)))
    logging.info('')
    return best    


def readNetAndParams(dir, fname):
    net = NetworkReader.readFrom(dir+fname)
    paramsList = net._unknown_argdict['HoF_PARAMS']
    scoreList = net._unknown_argdict['RUNRES']
    all = zip(scoreList, paramsList)
    return net, all

def readElitistNetAndParams(dir, fname):
    """determine if elitism kept multple copies in the HoF, and collapse them """
    net, all = readNetAndParams(dir, fname)
    gens = {}
    for i, (s, p) in enumerate(all):
        t = tuple(list(p))
        if t not in gens:
            gens[t] = [p, [i], [s]]
        else:
            gens[t][1].append(i)
            gens[t][2].append(s)
    return net, gens
    

def getBestAgent2(dir, fname, winningindex = 0):
    """ same, but reading from a net-xml file """
    net, all = readNetAndParams(dir, fname)
    all = zip(all, range(len(all)))
    all.sort(key = lambda x: -x[0][0])
    i = min(len(all)-1, winningindex)
    ((bval, best), gen) = all[i]
    logging.info('')
    logging.info('Trained on size: '+str(net.size))
    logging.info('Gen: '+str(net.size))
    logging.info('Score: '+str(bval))
    logging.info('Item '+str(i+1)+' of '+str(len(all)))
    return best, net.size


if __name__ == "__main__":
    from pybrain.rl.environments.twoplayergames import CaptureGame
    from pybrain.rl.agents.capturegameplayers import KillingPlayer, ModuleDecidingPlayer
    from pybrain.structure.networks.custom.capturegame import CaptureGameNetwork
    from resultreader import getTaggedFiles, selectSome
    
    playsize = 9
    steps = False
    learned = True
    #indices = range(200)
    indices = range(5)
    
    if not learned:
        size = None
        dkey = (5,1)
        dir = '../temp/stats/'
        fname = 'x-comparative-capture-killer'
    else:
        dir = '../temp/capturegame/1/'
        tag = 'p'
        ext = '.xml'
        files = getTaggedFiles(dir, tag, ext)
        selected = selectSome(files, [
                                      '7004',
                                      #'8283',
                                      ], requireAll = True)
        fname = selected[0][len(dir):]
        
    net, gens = readElitistNetAndParams(dir, fname)
    trainsize = net.size
    gv = map(lambda (p, g, s): (sum(s)/float(len(s)), g, p), gens.values())
    gv.sort(key = lambda x: -x[0])
    for s, g, p in gv:
        print s, g
        if len(p) < 20:
            print ' '*5, fListToString(p, 3)
        
    cheatsheet = {5: 1, 12:2, 21:3, 45:5, 140:10}
    g = CaptureGame(playsize)
    p1 = KillingPlayer(g)
    
    if learned:
        logging.basicConfig(level = logging.INFO, 
                            format = '%(message)s',
                            filename = dir+'playexamples'+fname[-10:-4]+'.txt', 
                            filemode = 'w')
    else:
        logging.basicConfig(level = logging.INFO, 
                            format = '%(message)s',
                            filename = dir+'playexamples.txt', 
                            filemode = 'w')
    
    logging.info('Players from file: '+dir+fname)
    logging.info('')
            
    for windex in indices:
        print windex
        if learned:
            w = gv[windex][2]
            logging.info('Trained Size: '+str(trainsize))
            logging.info('Score: '+str(gv[windex][0]))
            logging.info('Item '+str(windex)+' of '+str(len(gv)))            
        else:
            w = getBestAgent(dir, fname, size, dkey, windex)
        
        if w == None:
            print 'No agent found.'
        else:
            hsize = cheatsheet[w.size]
            if learned and trainsize != playsize:
                res = randEval(size = playsize, hsize = hsize, setParams = w, avgOver = 40, verbose = False)
                logging.info('Score on Size '+str(playsize)+':'+str(res))            
            net = CaptureGameNetwork(size = playsize, hsize = hsize, simpleborders = True)
            net._params[:] = w
            logging.info(fListToString(net.params, 3))
            p2 = ModuleDecidingPlayer(net, g, greedySelection = True)        
            arena(g, p1, p2, steps)
    
