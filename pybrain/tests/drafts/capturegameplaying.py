__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.tasks.capturegame import CaptureGameTask, HandicapCaptureTask
from pybrain.structure.evolvables.cheaplycopiable import CheaplyCopiable
from pybrain.rl.learners import ES
from pybrain.utilities import storeCallResults, fListToString
from pybrain.rl.agents.capturegameplayers import KillingPlayer, ClientCapturePlayer
from pybrain.structure.modules.mdlstm import MDLSTMLayer
from pybrain.tools.xml import NetworkWriter
import pylab
from scipy import array, average
    
# task settings: opponent, averaging to reduce noise, board size, etc.
size = 5
hsize = 10
evals = 100
avgover = 250
population = 2
mutationStd = 0.05

dir = '../temp/capturegame/3/'
tag = 'x0-'
storage = True
plotting = True


if False:
    javaTask = CaptureGameTask(size, averageOverGames = avgover, opponent = ClientCapturePlayer)
    class javaEval:        
        def __call__(self, p):
            javaTask.opponent.randomPartMoves = 0.2
            res = None
            while res == None:
                try:
                    res = javaTask(p)
                except:
                    print 'Oh-oh.'
            return res

task = CaptureGameTask(size, averageOverGames = avgover, opponent = KillingPlayer, numMovesCoeff = 0.2)
#task = HandicapCaptureTask(size, opponent = KillingPlayer, minEvals = 10)
#task = javaEval()

# keep track of evaluations for plotting
res = storeCallResults(task)

if False:
    # simple network
    from pybrain import buildNetwork, SigmoidLayer
    net = buildNetwork(task.outdim, task.indim, outclass = SigmoidLayer)
else:
    # specialized mdrnn variation
    from pybrain.structure.networks.custom.capturegame import CaptureGameNetwork
    net = CaptureGameNetwork(size = size, hsize = hsize, simpleborders = True, #componentclass = MDLSTMLayer
                             )
net.mutationStd = mutationStd
net._params /= 10
net = CheaplyCopiable(net)
print net.name, 'has', net.paramdim, 'trainable parameters.'
assert population % 2 == 0
learner = ES(task, net, mu = population/2, lambada = population/2, verbose = True, noisy = True)


fname = net.name[:-5]+'-'+str(learner)
fname += '-avg'+str(avgover)
fname += '-mut'+str(mutationStd)
    
print fname

newnet, f = learner.learn(evals)

if plotting:
    # plot the progression
    bests = []
    avgs = []
    b = population
    for i in range(len(res)/b):
        bests.append(max(res[i*b:(i+1)*b]))
        avgs.append(average(res[i*b:(i+1)*b]))
    print len(res),len(bests)
    print 'best per generation', fListToString(bests, 2)
    print 'avg per generation', fListToString(avgs, 2)
    pylab.plot(res, '.')
    pylab.plot(b/2+ b*array(range(len(bests))), bests, '-', label = 'max')
    pylab.plot(b/2+ b*array(range(len(bests))), avgs, '-', label = 'avg')
    pylab.legend()
    pylab.title(fname)

# store result
if storage:
    n = newnet.getBase()
    n.argdict['RUNRES'] = res[:]
    ps = []
    for h in learner.hallOfFame:
        ps.append(h.params.copy())
    n.argdict['HoF_PARAMS'] = ps
    
    NetworkWriter.writeToFile(n, dir+tag+fname+'.xml')

if False:
    # now, let's take the result, and compare it's performance on a larger game-baord
    newsize = 9
    bignew = newnet.getBase().resizedTo(newsize)
    bigold = net.getBase().resizedTo(newsize)

    newtask = CaptureGameTask(newsize, averageOverGames = 100, opponent = KillingPlayer)
    print 'Old net on medium board score:', newtask(bigold)
    print 'New net on medium board score:', newtask(bignew)

if False:
    # now, let's take the result, and compare it's performance on an even larger game-baord
    newsize = 19
    bignew = newnet.getBase().resizedTo(newsize)
    bigold = net.getBase().resizedTo(newsize)

    newtask = CaptureGameTask(newsize, averageOverGames = 100, opponent = KillingPlayer)
    print 'Old net on big board score:', newtask(bigold)
    print 'New net on big board score:', newtask(bignew)

if plotting:
    pylab.show()