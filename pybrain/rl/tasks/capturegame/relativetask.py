__author__ = 'Tom Schaul, tom@idsia.ch'

from capturetask import CaptureGameTask
from pybrain.rl.experiments import EpisodicExperiment
from pybrain.rl.tasks import EpisodicTask
from pybrain.rl.agents.capturegameplayers import ModuleDecidingPlayer
from pybrain.rl.environments.twoplayergames import CaptureGame
from pybrain.rl.agents.capturegameplayers.captureplayer import CapturePlayer


class RelativeCaptureTask(CaptureGameTask):
    """ returns the (anti-symmetric) relative score of p1 with respect to p2.
    (p1 and p2 are CaptureGameNetworks)
    The score depends on: 
    - greedy play
    - play with fixed starting positions (only first stone)
    - moves-until-win or moves-until-defeat (winning faster is better)
    - play with noisy moves (e.g. adjusting softmax temperature)
        
    """
    
    # are networks provided?
    useNetworks = False
    
    # 
    
    def __init__(self, size, **args):
        self.setArgs(**args)
        self.size = size
        self.task = CaptureGameTask(self.size)
        self.env = self.task.env

    def __call__(self, p1, p2):
        if self.useNetworks:
            p1 = ModuleDecidingPlayer(p1, self.task.env, greedySelection = False, temperature = 0)
            p2 = ModuleDecidingPlayer(p2, self.task.env, greedySelection = False, temperature = 0)
        else:
            assert isinstance(p1, CapturePlayer)
            assert isinstance(p2, CapturePlayer)
        p1.color = CaptureGame.BLACK
        p2.color = -p1.color
        self.player = p1
        self.opponent = p2
        
        res = 0
        # one greedy game
        res += self._oneGame()
        
        return res
    
    def _setTemperature(self, t):
        if self.useNetworks:
            self.opponent.temperature = t
            self.player.temperature = t
        elif hasattr(self.opponent, 'randomPartMoves'):
            # an approximate conversion of temperature into random proportion:
            randPart = t/(t+1)
            self.opponent.randomPartMoves = randPart
            self.player.randomPartMoves = randPart
    
    def _fixedStartingPos(self):
        """ a list of starting positions, not along the border, and respecting symmetry. """
        res = []
        if self.size < 3:
            return res
        for x in range(1, (self.size+1)/2):
            for y in range(x, (self.size+1)/2):
                res.append((x,y))
        return res
            
    def _oneGame(self):
        allRs = EpisodicExperiment(self.task, self.player).doEpisodes(1)
        print len(allRs), allRs
        return self.getTotalReward()
        steps = 0
        self.player.newEpisode()
        while not self.task.isFinished():
            steps += 1
            self.player.integrateObservation(self.task.getObservation())
            self.task.performAction(self.player.getAction())
            reward = self.task.getReward()
            self.player.giveReward(reward)
        print steps, reward
        return self.task.getTotalReward()
        
    
if __name__ == '__main__':
    assert RelativeCaptureTask(5)._fixedStartingPos() == [(1, 1), (1, 2), (2, 2)]
    assert RelativeCaptureTask(8)._fixedStartingPos() == [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
    r = RelativeCaptureTask(5)
    from pybrain.rl.agents.capturegameplayers import RandomCapturePlayer
    p1 = RandomCapturePlayer(r.env)
    p2 = RandomCapturePlayer(r.env)
    print r(p1, p2)
    print r.env
    