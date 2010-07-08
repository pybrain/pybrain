__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.utilities import abstractMethod
from pybrain.rl.environments import Environment


class CompetitiveEnvironment(Environment):
    """ an environment in which multiple agents interact, competitively.
    This class is only for conceptual grouping, it only constrains the format of action input.
    """

    def performAction(self, action):
        """ perform an action on the world that changes it's internal state (maybe stochastically)

            :key action: an action that should be executed in the Environment, by an agent.
            :type action: tuple: (agentID, action value)
            :note: This function is abstract and has to be implemented.
        """
        abstractMethod()


class TwoPlayerGame(CompetitiveEnvironment):
    """ a game between 2 players, alternating turns. Outcome can be one winner or a draw. """

    DRAW = 'draw'

    def reset(self):
        self.winner = None
        self.lastplayer = None

    def performAction(self, action):
        self.lastplayer = action[0]
        self.doMove(*action)

    def doMove(self, player, action):
        """ the core method to be implemented bu all TwoPlayerGames:
        what happens when a player performs an action. """
        abstractMethod()

    def isLegal(self, player, action):
        """ is this a legal move? By default, everything is allowed. """
        return True

    def gameOver(self):
        """ is the game over? """
        return self.winner != None

    def getWinner(self):
        """ returns the id of the winner, 'draw' if it's a draw, and None if the game is undecided. """
        return self.winner
