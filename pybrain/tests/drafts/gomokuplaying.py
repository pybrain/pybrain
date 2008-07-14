__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.environments.twoplayergames import GomokuGame
from pybrain.rl.agents.gomokuplayers import RandomGomokuPlayer, KillingGomokuPlayer


if __name__ == '__main__':
    g = GomokuGame((7,7))
    p1, p2 = RandomGomokuPlayer(g), KillingGomokuPlayer(g)
    p2.color = -p1.color
    g.playToTheEnd(p1, p2)
    print g
    
