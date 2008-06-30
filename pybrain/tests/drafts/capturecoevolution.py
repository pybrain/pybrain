""" A little test for comptitive coevolution on the capturegame:
 dimension 4, with a simple net (3 hidden neurons)
 all evaluations are greedy (which is not good)."""

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.tasks.capturegame import CaptureGameTask
from pybrain.structure.evolvables.cheaplycopiable import CheaplyCopiable
from pybrain.structure.networks.custom.capturegame import CaptureGameNetwork
from pybrain.rl.learners.search.competitivecoevolution import CompetitiveCoevolution
from pybrain.rl.agents.capturegameplayers import KillingPlayer, ModuleDecidingPlayer

size = 4
hsize = 1
popsize = 8

# total games to be played:
evals = 640

net = CaptureGameNetwork(size = size, hsize = hsize, simpleborders = True, #componentclass = MDLSTMLayer
                         )
net._params /= 20
net = CheaplyCopiable(net)
print net.name, 'has', net.paramdim, 'trainable parameters.'

absoluteTask = CaptureGameTask(size, averageOverGames = 40, alternateStarting = True,
                               opponent = KillingPlayer)

def handicapTask(p1):
    """ The score for this task is not the percentage of wins, but the achieved handicap against
    killingplayer when the results stabilize. 
    Stabilize: if after minimum of 5 games at the same handicap H, > 80% were won by the player, increase the handicap. 
    if <40% decrease (this tends to not overestimate the handicap).
    If the system fluctuates between H and H+1, with at least 10 games played on each level,
    assert H+0.5 as handicap.
    the score = 2 * #handicaps + proportion of wins at that level. """
    
    # the maximal handicap given is a full line of stones along the second line.
    maxHandicaps = (size-2)*2+(size-4)*2
    
    # stores [wins, total] for each handicap-key
    results = {0: [0,0]}
    
    def winProp(h):
        w, t = results[h]
        if t > 0:
            return w/float(t)
        else:
            return 0.5
    
    def goUp(h):
        """ ready to go up one handicap? """
        if results[h][1] >= 5:
            return winProp(h) > 0.8
        return False
    
    def goDown(h):
        """ have to go down one handicap? """
        if results[h][1] >= 5:
            return winProp(h) < 0.4
        return False
    
    def bestHandicap():
        return max(results.keys())-1 
    
    def fluctuating():
        """ Is the highest handicap unstable? """
        high = bestHandicap()
        if high > 0:
            if results[high][1] > 10 and results[high-1][1] > 10:
                return goUp(high-1) and goDown(high)
        return False
    
    def stable(h):
        return (fluctuating() 
                or (results[h][1] > 10 and (not goUp(h)) and (not goDown(h)))
                or (results[h][1] > 10 and goUp(h) and h >= maxHandicaps))
    
    def addResult(h, win):
        if h+1 not in results:
            results[h+1] = [0,0]
        results[h][1]+=1
        if win > 0: 
            results[h][0]+=1
            
    
    task = CaptureGameTask(size, averageOverGames = 1, opponent = KillingPlayer)
    
    # main loop
    current = 0
    while not stable(current):
        task.reset()
        task.giveHandicap(current)
        addResult(current, task(p1))
        if goUp(current) and current < maxHandicaps:
            current += 1
        elif goDown(current) and current > 1:
            current -= 1
    high = bestHandicap()    
    if not fluctuating():
        return high*2 + winProp(high)
    else:
        return high*2 -1 + 0.5(winProp(high)+winProp(high-1))

def relativeTask(p1, p2):
    tmp = CaptureGameTask(size, averageOverGames = 1)
    tmp.opponent = ModuleDecidingPlayer(p2, tmp.env, greedySelection = True)
    player = ModuleDecidingPlayer(p1, tmp.env, greedySelection = True)
    res = tmp(player)
    return res

learner = CompetitiveCoevolution(relativeTask, net.copy(), net.copy(), populationSize = popsize, verbose = True)
newnet = learner.learn(evals)

if True:
    res = []
    for h in learner.hallOfFame:
        res.append(absoluteTask(h))
    res1 = []
    for h in learner.hallOfFame:
        res1.append(handicapTask(h))
    # plot the progression
    import pylab
    pylab.plot(res)
    pylab.plot(res1)
    pylab.title(net.name)

if False:
    # now, let's take the result, and compare it's performance on a larger game-baord
    newsize = 9
    bignew = newnet.getBase().resizedTo(newsize)
    bigold = net.getBase().resizedTo(newsize)

    newtask = CaptureGameTask(newsize, averageOverGames = 100, alternateStarting = True,
                              opponent = KillingPlayer)
    print 'Old net on medium board score:', newtask(bigold)
    print 'New net on medium board score:', newtask(bignew)

pylab.show()