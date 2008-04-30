__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl import Agent


class CapturePlayer(Agent):
    """ a class of agent that can play the capture game, i.e. provides actions in the format:
    (playerid, position)
    playerid is self.color, by convention. 
    It generally also has access to the game object. """
    def __init__(self, game, color):
        self.game = game
        self.color = color
    
    