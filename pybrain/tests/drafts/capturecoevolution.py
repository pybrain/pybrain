""" A little test for comptitive coevolution on the capturegame. """

__author__ = 'Tom Schaul, tom@idsia.ch'

import pylab
    
from pybrain.tools.plotting.ciaoplot import CiaoPlot    
from pybrain.structure.evolvables.cheaplycopiable import CheaplyCopiable
from pybrain.structure.networks.custom.capturegame import CaptureGameNetwork
from pybrain.rl.tasks.capturegame import CaptureGameTask, HandicapCaptureTask, RelativeCaptureTask
from pybrain.rl.agents.capturegameplayers.clientwrapper import ClientCapturePlayer
from pybrain.rl.learners.search.competitivecoevolution import CompetitiveCoevolution
from pybrain.rl.learners.search.coevolution import Coevolution
from pybrain.rl.agents.capturegameplayers import KillingPlayer
from pybrain.tools.xml import NetworkWriter
    
# parameters
size = 5
hsize = 5
popsize = 8
generations = 50
elitist = True
temperature = 0.05 # for learning games
relTaskAvg = 1
hallOfFameProp = 0.5
selProp = 0.25
beta = 1
tournSize = 4
absProp = 0.
mutationStd = 0.01
competitive = False

# experiment settings
ciao = False
absplot = True
scalingtest = False
storage = True
javaTest = False

# total games to be played:
if tournSize != None:
    onegeneration = tournSize * popsize * 2
elif competitive:
    onegeneration = popsize*popsize*2
else:
    onegeneration = popsize*(popsize-1)*2
    
evals = generations * onegeneration

net = CaptureGameNetwork(size = size, hsize = hsize, simpleborders = True, #componentclass = MDLSTMLayer
                         )
net.mutationStd = mutationStd
net._params /= 20
net = CheaplyCopiable(net)
print net.name, 'has', net.paramdim, 'trainable parameters.'

absoluteTask = CaptureGameTask(size, averageOverGames = 40, alternateStarting = True, opponent = KillingPlayer)
handicapTask = HandicapCaptureTask(size, opponent = KillingPlayer)
relativeTask = RelativeCaptureTask(size, useNetworks = True)
    
def buildName():
    name = ''
    if competitive:
        name += 'Comp'
    name += 'Coev'
    if relTaskAvg > 1:
        name += '-rA'+str(relTaskAvg)
    if elitist:
        name += '-elit'
    name += '-T'+str(temperature)
    name += '-e'+str(evals)
    name += '-pop'+str(popsize)
    if tournSize != None:
        name += '-tSize'+str(tournSize)
    if beta < 1:
        name += '-pc_avg'+str(beta)
    if hallOfFameProp > 0:
        name += '-HoF'+str(hallOfFameProp)
    if selProp != 0.5:
        name += '-selP'+str(selProp)
    if absProp > 0:
        name += '-absP'+str(absProp)
    if mutationStd != 0.1:
        name += '-mut'+str(mutationStd)
    name += net.name[19:-5]
    return name

name = buildName()
print name
    
res = []
hres = []   
jres = []

if competitive: 
    lclass = CompetitiveCoevolution
else:
    lclass = Coevolution

seeds = []
for dummy in range(popsize):
    tmp = net.copy()
    tmp.randomize()
    seeds.append(tmp)

learner = lclass(lambda x,y: relativeTask(x,y, temperature), 
                 seeds, 
                 elitism = elitist, 
                 parentChildAverage = beta,
                 tournamentSize = tournSize,
                 populationSize = popsize, 
                 selectionProportion = selProp,
                 hallOfFameEvaluation = hallOfFameProp,
                 absEvalProportion = absProp,
                 absEvaluator = absoluteTask,
                 verbose = True)
if javaTest:
    try:
        javaTask = CaptureGameTask(size, averageOverGames = 40, alternateStarting = True,
                                   opponent = ClientCapturePlayer)
        javaTask.opponent.randomPartMoves = 0.2
    except:
        print 'No server found.'
        javaTest = False
    
for g in range(generations):
    newnet = learner.learn(onegeneration)
    h = learner.hallOfFame[-1]
    res.append(absoluteTask(h))
    hres.append(handicapTask(h))
    print res[-1], hres[-1], '(evals:', learner.steps, ')'
    if javaTest:
        try:
            jres.append(javaTask(h))
            print 'Java-play:', jres[-1]
        except:
            jres.append(0)
            print 'Server playing error.'
        
# store result
if storage and evals > 100 and size > 3:
    n = newnet.getBase()
    n.argdict['RUNRES'] = res[:]
    n.argdict['RUNRESH'] = hres[:]
    n.argdict['RUNRESJ'] = jres[:]
    ps = []
    for h in learner.hallOfFame:
        ps.append(h.params.copy())
    n.argdict['HoF_PARAMS'] = ps
    NetworkWriter.writeToFile(n, '../temp/capturegame/'+name)

# plot CIAO diagram
if ciao:    
    hof = learner.hallOfFame
    if competitive and generations % 2 == 0:
        hof1 = hof[0::2]
        hof2 = hof[1::2]
    else:
        hof1 = hof
        hof2 = hof
    p = CiaoPlot(lambda x,y: relativeTask(x, y, 0), hof1, hof2)
    pylab.title('CIAO'+name)
    
if absplot:
    # plot the progression
    pylab.figure()
    pylab.plot(res)
    pylab.plot(hres)
    if javaTest:
        pylab.plot(jres)
    pylab.title(name)
    
if scalingtest:
    # now, let's take the result, and compare its performance on a larger game-baord
    newsize = 9
    bignew = newnet.getBase().resizedTo(newsize)
    bigold = net.getBase().resizedTo(newsize)

    newtask = CaptureGameTask(newsize, averageOverGames = 100, alternateStarting = True,
                              opponent = KillingPlayer)
    print 'Old net on medium board score:', newtask(bigold)
    print 'New net on medium board score:', newtask(bignew)

if ciao:
    p.show()
elif absplot:
    pylab.show()