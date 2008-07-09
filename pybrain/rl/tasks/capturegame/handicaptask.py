__author__ = 'Tom Schaul, tom@idsia.ch'

from capturetask import CaptureGameTask

# TODO: parametrize hard-coded variables.
# TODO: also allow handicap-advantage

class HandicapCaptureTask(CaptureGameTask):
    """ Play against an opponent, and try to beat it with it having the maximal 
    number of handicap stones: 
    
    The score for this task is not the percentage of wins, but the achieved handicap against
    the opponent when the results stabilize. 
    Stabilize: if after minimum of 5 games at the same handicap H, > 80% were won by the player, increase the handicap. 
    if <40% decrease (this tends to not overestimate the handicap).
    If the system fluctuates between H and H+1, with at least 10 games played on each level,
    assert H+0.5 as handicap.
    the score = 2 * #handicaps + proportion of wins at that level. """
    
    maxGames = 200
    averageOverGames = 1
    
    def __init__(self, *args, **kargs):
        CaptureGameTask.__init__(self, *args, **kargs)
        self.size = self.env.size
        # the maximal handicap given is a full line of stones along the second line.
        self.maxHandicaps = (self.size-2)*2+(self.size-4)*2
            
    def winProp(self, h):
        w, t, dummy, dummy = self.results[h]
        if t > 0:
            return w/float(t)
        else:
            return 0.5
    
    def goUp(self, h):
        """ ready to go up one handicap? """
        if self.results[h][1] >= 5:
            return self.winProp(h) > 0.8
        return False
    
    def goDown(self, h):
        """ have to go down one handicap? """
        if self.results[h][1] >= 5:
            return self.winProp(h) < 0.4
        return False
    
    def bestHandicap(self):
        return max(self.results.keys())-1 
    
    def fluctuating(self):
        """ Is the highest handicap unstable? """
        high = self.bestHandicap()
        if high > 0:
            if self.results[high][1] > 10 and self.results[high-1][1] > 10:
                return self.goUp(high-1) and self.goDown(high)
        return False
    
    def stable(self, h):
        return (self.fluctuating() 
                or (self.results[h][1] > 10 and (not self.goUp(h)) and (not self.goDown(h)))
                or (self.results[h][1] > 10 and self.goUp(h) and h >= self.maxHandicaps)
                or (self.results[h][1] > 10 and self.goDown(h) and h == 0))
    
    def addResult(self, h, win, moves):
        if h+1 not in self.results:
            self.results[h+1] = [0,0,0,0]
        self.results[h][1] += 1
        if win == True: 
            self.results[h][0] += 1
            self.results[h][2] += moves            
        else:
            self.results[h][3] += moves
    
    def reset(self):
        # stores [wins, total, sum(moves-til-win), sum(moves-til-lose)] 
        # for each handicap-key
        self.results = {0: [0,0,0,0]}
            
    def __call__(self, player):            
        self.reset()
        current = 0
        games = 0
        while games < self.maxGames and not self.stable(current):
            games += 1
            self.env.reset()
            self.env.giveHandicap(current)
            self.env._playToTheEnd(self.opponent, player)
            win = self.env.winner == player.color
            print self.env
            self.addResult(current, win, self.env.movesDone)
            if self.goUp(current) and current < self.maxHandicaps:
                current += 1
            elif self.goDown(current) and current > 1:
                current -= 1
            
        high = self.bestHandicap()    
        if not self.fluctuating():
            return high*0.5 + 2*self.winProp(high) -1
        else:
            return high*0.5 -0.25 + 0.5 * 2* (self.winProp(high)+self.winProp(high-1)) -1
        
if __name__ == '__main__':
    from pybrain.rl.agents.capturegameplayers import RandomCapturePlayer, ClientCapturePlayer
    h = HandicapCaptureTask(5, opponentStart = False)
    p1 = RandomCapturePlayer(h.env)
    p2 = ClientCapturePlayer(h.env, verbose = True)
    h.reset()
    h.env.giveHandicap(3)
    print h.env
    print p2.color
    print h.opponent.color
    h.env._playToTheEnd(p2, h.opponent)
    print h.env        
    