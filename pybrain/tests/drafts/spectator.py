""" A little tool for watching games """

__author__ = 'Tom Schaul, tom@idsia.ch'

import sys
import logging

from pybrain.utilities import fListToString
from nesexperiments import pickleReadDict

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
    logging.info('Training Size: '+str(size))
    logging.info('HiddenSize: '+str(dkey[0])+' initScaling: '+str(dkey[1]))
    logging.info('Score: '+str(bval))
    logging.info('Item '+str(i+1)+' of '+str(len(all)))
    logging.info('')
    return best    


if __name__ == "__main__":
    from pybrain.rl.environments.twoplayergames import CaptureGame
    from pybrain.rl.agents.capturegameplayers import KillingPlayer, ModuleDecidingPlayer
    from pybrain.structure.networks.custom.capturegame import CaptureGameNetwork
    
    size = None
    playsize = 11
    dkey = (5,1)
    steps = False
    
    dir = '../temp/stats/'
    fname = 'x-comparative-capture-killer'
    cheatsheet = {5: 1, 12:2, 45:5}
    g = CaptureGame(playsize)
    p1 = KillingPlayer(g)
    logging.basicConfig(level = logging.INFO, 
                        format = '%(message)s',
                        filename = dir+'playexamples.txt', 
                        filemode = 'w')
    logging.info('Players from file: '+dir+fname)
    logging.info('')
            
    for windex in range(0,200):
        print windex
        w = getBestAgent(dir, fname, size, dkey, windex)
        #w = [ -0.0765, -0.0673,  0.0050, -0.1023,  0.1692]
        if w == None:
            print 'No agent found.'
        else:
            hsize = cheatsheet[w.size]
            net = CaptureGameNetwork(size = playsize, hsize = hsize, simpleborders = True)
            net._params[:] = w
            logging.info(fListToString(net.params, 3))
            p2 = ModuleDecidingPlayer(net, g, greedySelection = True)        
            arena(g, p1, p2, steps)
    
