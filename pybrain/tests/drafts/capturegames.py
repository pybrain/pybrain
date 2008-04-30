""" play a hundred random capture games """

__author__ = 'Tom Schaul, tom@idsia.ch'

    
from pybrain.rl.environments.twoplayergames import CaptureGame
from random import choice


if __name__ == '__main__':
    c = CaptureGame(5)
    
    moves = 0
        
    for g in range(100):
        c.reset()
        player = 1
        while c.winner == None:
            p = c.getAcceptable(player)
            if len(p) == 0:
                p = c.getLegals(player)
            pos = choice(p)
            c.performAction([player, pos])
            player = -player
            moves += 1
        print c, 'Moves', moves