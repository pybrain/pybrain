""" A little tool for watching games """

__author__ = 'Tom Schaul, tom@idsia.ch'

import sys

from pybrain.utilities import fListToString


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
                    print fListToString(o[s*j:s*(j+1)], 4)
    if not steps:
        print game


if __name__ == "__main__":
    from pybrain.rl.environments.twoplayergames import CaptureGame
    from pybrain.rl.agents.capturegameplayers import KillingPlayer, ModuleDecidingPlayer
    from pybrain.structure.networks.custom.capturegame import CaptureGameNetwork
    
    size = 7
    w = [ -0.0765, -0.0673,  0.0050, -0.1023,  0.1692]
    
    g = CaptureGame(size)
    p1 = KillingPlayer(g)
    net = CaptureGameNetwork(size = size, hsize = len(w)/5, simpleborders = True)
    net._params[:] = w
    print net.params
    p2 = ModuleDecidingPlayer(net, g, greedySelection = True)
    
    arena(g, p1, p2)
    
