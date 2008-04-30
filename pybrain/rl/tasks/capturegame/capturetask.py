__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import sqrt

from pybrain.rl import EpisodicTask, EpisodicExperiment, LearningAgent
from pybrain.utilities import confidenceIntervalSize
from pybrain.rl.environments.twoplayergames import CaptureGame
from pybrain.rl.agents.capturegameplayers import RandomCapturePlayer, NonSuicidePlayer


class CaptureGameTask(EpisodicTask):
    
    def __init__(self, size, scoretoreward = 1, suicide = True, opponentstart = False):        
        EpisodicTask.__init__(self, CaptureGame(size))
        self.scoretoreward = scoretoreward
        if opponentstart:
            oppcolor = CaptureGame.BLACK
        else:
            oppcolor = CaptureGame.WHITE
            
        if suicide:
            self.opponent = RandomCapturePlayer(self.env, oppcolor)
        else:
            self.opponent = NonSuicidePlayer(self.env, oppcolor)
        
        self.reset()
                    
    def isFinished(self):
        return self.env.gameOver()
    
    def getReward(self):
        if self.isFinished():
            return - self.opponent.color * self.env.winner * self.scoretoreward
        else:
            return 0
        
    def performAction(self, action):
        EpisodicTask.performAction(self, action)
        if not self.isFinished():
            EpisodicTask.performAction(self, self.opponent.getAction())
                
    def reset(self):
        EpisodicTask.reset(self)   
        if self.opponent.color == CaptureGame.BLACK:     
            # first move by opponent
            EpisodicTask.performAction(self, self.opponent.getAction())
            
    def alternateStarting(self, agent):
        tmp = self.opponent.color
        self.opponent.color = agent.color                
        agent.color = tmp
        
    def getPerformance(self, agent, precision = 0.1, maxruns = None):
        """ this task measures the percentage of games won against its opponent. """
        sum = 0.
        wins = 0.
        confintervalsize = 1.    
        if isinstance(agent, LearningAgent):
            agent.disableTraining()  
        while confintervalsize > precision:
            sum += 1            
            self.alternateStarting(agent)                
            self.reset()
            # do one game:
            E = EpisodicExperiment(self, agent)
            E.doEpisodes()
            if self.env.winner == agent.color:
                wins += 1
            if sum > 2/precision:
                stdev = sqrt(wins*(sum-wins)/(sum*(sum-1)))                
                confintervalsize = confidenceIntervalSize(stdev, sum)   
                if maxruns != None and maxruns <= sum:
                    break
        if isinstance(agent, LearningAgent):
            agent.enableTraining()          
            agent.newEpisode()
        mu = wins/sum              
        # round to something corresponding to the precision        
        mu = float(int(mu*(5/precision)))/(5/precision)
        return mu, stdev, sum, confintervalsize