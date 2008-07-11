__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.tasks.capturegame import CaptureGameTask, HandicapCaptureTask
from pybrain.structure.evolvables.cheaplycopiable import CheaplyCopiable
from pybrain.rl.learners import ES
from pybrain.utilities import storeCallResults
from pybrain.rl.agents.capturegameplayers import KillingPlayer, ClientCapturePlayer
from pybrain.structure.modules.mdlstm import MDLSTMLayer

# task settings: opponent, averaging to reduce noise, board size, etc.
size = 5
hsize = 2
evals = 400
avgover = 40


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

#task = CaptureGameTask(size, averageOverGames = avgover, opponent = KillingPlayer)
#task = HandicapCaptureTask(size, opponent = KillingPlayer, minEvals = 10)
task = javaEval()

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
net._params /= 10
net = CheaplyCopiable(net)
print net.name, 'has', net.paramdim, 'trainable parameters.'

learner = ES(task, net, mu = 5, lambada = 5, verbose = True, noisy = True)
newnet, f = learner.learn(evals)

# plot the progression
import pylab
pylab.plot(res)
pylab.title(net.name)

# store result
if True:
    from pybrain.tools.xml import NetworkWriter
    n = newnet.getBase()
    n.argdict['RUNRES'] = res[:]
    NetworkWriter.writeToFile(n, '../temp/capturegame/JN-e'+str(evals)+'-avg'+str(avgover)+newnet.name[18:-5])

if True:
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

pylab.show()