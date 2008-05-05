__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl import EpisodicTask
from inspect import isclass
from pybrain.utilities import  Named
from pybrain.rl.environments.twoplayergames import CaptureGame
from pybrain.rl.agents.capturegameplayers import RandomCapturePlayer, ModuleDecidingPlayer
from pybrain.rl.agents.capturegameplayers.captureplayer import CapturePlayer
from pybrain.structure.modules.module import Module


class CaptureGameTask(EpisodicTask, Named):
    """  """
    
    # first game, opponent is black
    opponentStart = True
    
    # on subsequent games, starting players are alternating
    alternateStarting = False    
    
    # numerical reward value attributed to winning
    winnerReward = 1.
    
    # average over some games for evaluations
    averageOverGames = 10
    
    noisy = True
    
    def __init__(self, size, opponent = None, **args):        
        EpisodicTask.__init__(self, CaptureGame(size))
        self.setArgs(**args)
        if opponent == None:
            opponent = RandomCapturePlayer(self.env)
        elif isclass(opponent):
            # assume the agent can be initialized without arguments then.
            opponent = opponent(self.env)
        if not self.opponentStart:
            opponent.color = CaptureGame.WHITE
        self.opponent = opponent
        self.reset()
                    
    def reset(self):
        self.switched = False
        EpisodicTask.reset(self)   
        if self.opponent.color == CaptureGame.BLACK:     
            # first move by opponent
            EpisodicTask.performAction(self, self.opponent.getAction())
    
    def isFinished(self):
        res = self.env.gameOver()
        if res and self.alternateStarting and not self.switched:
            # alternate starting player
            self.opponent.color *= -1       
            self.switched = True     
        return res
    
    def getReward(self):
        if self.isFinished():
            return - self.opponent.color * self.env.winner * self.winnerReward
        else:
            return 0
        
    def performAction(self, action):
        EpisodicTask.performAction(self, action)
        if not self.isFinished():
            EpisodicTask.performAction(self, self.opponent.getAction())
            
    def __call__(self, x):
        """ If a module is given, wrap it into a ModuleDecidingAgent before evaluating it. 
        Also, if applicable, average the result over multiple games. """
        if isinstance(x, Module):
            agent = ModuleDecidingPlayer(x, self.env, greedySelection = True)
        elif isinstance(x, CapturePlayer):
            agent = x
        else:
            raise NotImplementedError('Missing implementation for '+x.__class__.__name__+' evaluation')
        res = 0
        for dummy in range(self.averageOverGames):
            agent.color = -self.opponent.color
            res += EpisodicTask.__call__(self, agent)
        return res / float(self.averageOverGames)
        
        